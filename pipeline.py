from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import config
from models import db, AnalysisRun, Artifact, MetricResult
from biomechanics import (
    analyze,
    metrics_for_position,
    draw_upper_annotation,
)
from scoring_adapter import build_report
from overlay_video import create_skeleton_overlay

EXECUTOR = ThreadPoolExecutor(max_workers=1)


def probe_video_fps(video_path: Path) -> float:
    """Read the actual source-video frame rate with ffprobe."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rate = completed.stdout.strip()
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / float(denominator)
    else:
        fps = float(rate)

    if not 1.0 <= fps <= 240.0:
        raise ValueError(f"Invalid video frame rate reported by ffprobe: {fps}")

    return fps


def cmd(run_dir, stage, args, env=None):
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "stage": stage})
    )
    merged = os.environ.copy()
    merged.update(env or {})

    with (run_dir / "pipeline.log").open("a") as log:
        log.write(f"\n===== {stage} =====\n")
        log.flush()
        rc = subprocess.call(
            args,
            cwd=config.CODES_DIR,
            env=merged,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if rc:
        raise RuntimeError(f"{stage} failed. See the internal pipeline log.")


def add_artifact(run, run_dir, kind, path, name=None):
    path = Path(path)
    if not path.exists():
        return

    try:
        relative = path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        relative = Path("saved_artifacts") / kind / path.name
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
        path = target

    db.session.add(Artifact(
        analysis_run_id=run.id,
        artifact_type=kind,
        relative_path=str(path.resolve().relative_to(run_dir.resolve())),
        original_filename=name or path.name,
        file_size_bytes=path.stat().st_size if path.is_file() else None,
    ))




def validate_report_images(report: dict, run_dir: Path) -> None:
    """
    Fail early when the report references an annotation that was not
    actually created. This prevents a completed report with empty image
    cards.
    """
    missing = []

    for cycle in report.get("cycles", []):
        cycle_id = cycle.get("gait_cycle")
        for position, pdata in cycle.get("positions", {}).items():
            if pdata.get("missing"):
                continue

            event_frame = pdata.get("event_frame")
            frame_data = pdata.get("frames", {}).get(str(event_frame), {})

            raw_image = frame_data.get("raw_image")
            if raw_image and not (run_dir / raw_image).exists():
                missing.append(raw_image)

            for result in frame_data.get("metrics", {}).values():
                annotated = result.get("annotated_image")
                if annotated and not (run_dir / annotated).exists():
                    missing.append(annotated)

            for result in pdata.get("upper_body_metrics", []):
                annotated = result.get("annotated_image")
                if annotated and not (run_dir / annotated).exists():
                    missing.append(annotated)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Annotated report images were not created: {preview}"
        )


def submit(app, run_id):
    EXECUTOR.submit(execute, app, run_id)


def recover_interrupted_runs():
    """
    Jobs live in this process's in-memory executor, so a server restart loses
    them. Mark those runs failed so they do not show "running" forever. A
    failed run does not block re-uploading the same video.
    """
    interrupted = AnalysisRun.query.filter(
        AnalysisRun.status.in_(["queued", "running"])
    ).all()
    for run in interrupted:
        run.status = "failed"
        run.stage = "Analysis failed"
        run.error_message = (
            "The server restarted before this analysis finished. "
            "Upload the video again to rerun it."
        )
    if interrupted:
        db.session.commit()


def pipeline_key(run) -> str:
    """
    Name used for this run inside the shared external-pipeline folders
    (sprint_videos, rollout results, exported JSON). It includes part of the
    run id so two videos of the same athlete never share, overwrite, or
    reuse (via SKIP_EXISTING) each other's reconstruction outputs.
    """
    return f"{run.player.athlete_name}_{run.public_id[:8]}"


def analyze_saved_run(run, run_dir: Path, source_fps: float, key: str) -> None:
    """
    Analysis half of the pipeline: event detection, metrics, annotations,
    overlay video, report.json, and database rows.

    Uses only files already inside run_dir (frames/, datasets/lowerbody.json,
    datasets/smpl.json) plus the scoring core, so it needs no GPU and can
    re-analyze a saved run after thresholds or detection logic change.
    The caller commits the session.
    """
    athlete = run.player.athlete_name
    video = run_dir / "input" / "original_video.mp4"
    frames = run_dir / "frames"
    saved_smpl = run_dir / "datasets" / "smpl.json"
    saved_lower = run_dir / "datasets" / "lowerbody.json"

    report = build_report(athlete, saved_lower, frames)
    bio = analyze(saved_smpl, saved_lower)

    # Record frame-count alignment instead of silently assuming that
    # every extracted image and every JSON sequence has equal length.
    lower_data = json.loads(saved_lower.read_text())
    smpl_data = json.loads(saved_smpl.read_text())
    joint_lengths = {
        name: len(values.get("pixel_uv", []))
        for name, values in lower_data.get("joints", {}).items()
        if "pixel_uv" in values
    }
    extracted_count = len(list(frames.glob("*.jpg")))
    lower_min = min(joint_lengths.values()) if joint_lengths else 0
    lower_max = max(joint_lengths.values()) if joint_lengths else 0
    smpl_count = int(smpl_data.get("n_frames", len(smpl_data.get("frames", []))))
    warnings = []
    if lower_min != lower_max:
        warnings.append(
            f"Lower-body joint arrays have different lengths ({lower_min}–{lower_max})."
        )
    if len({extracted_count, lower_min, smpl_count}) != 1:
        warnings.append(
            "Extracted images, lower-body JSON, and SMPL JSON do not have matching frame counts."
        )
    report["alignment"] = {
        "extracted_image_count": extracted_count,
        "lowerbody_json_frame_count": lower_min,
        "lowerbody_joint_min": lower_min,
        "lowerbody_joint_max": lower_max,
        "smpl_json_frame_count": smpl_count,
        "source_fps": source_fps,
        "warnings": warnings,
    }
    report["pipeline_key"] = key

    # Add position-specific upper-body mechanics and annotated images.
    frame_files = sorted(frames.glob("*.jpg"))
    annotation_dir = run_dir / "annotations" / "upper_body"

    for cycle in report["cycles"]:
        cycle_id = int(cycle["gait_cycle"])
        for position, pdata in cycle["positions"].items():
            if pdata.get("missing") or pdata.get("event_frame") is None:
                continue

            event_frame = int(pdata["event_frame"])
            if event_frame >= len(bio["frames"]):
                warnings.append(
                    f"Cycle {cycle_id} {position}: frame {event_frame} is "
                    "beyond the SMPL data, so upper-body metrics were skipped."
                )
                pdata["upper_body_metrics"] = []
                continue

            upper = metrics_for_position(
                bio["frames"][event_frame],
                cycle_id,
                position,
            )

            source = (
                frame_files[event_frame]
                if event_frame < len(frame_files)
                else None
            )

            for result in upper:
                if source:
                    destination = (
                        annotation_dir
                        / f"cycle_{cycle_id:02d}_{position}_"
                          f"{result['metric_id']}.jpg"
                    )
                    draw_upper_annotation(
                        source,
                        destination,
                        result,
                        bio["frames"][event_frame],
                        bio["joint_names"],
                    )
                    result["annotated_image"] = str(
                        destination.relative_to(run_dir)
                    )

            pdata["upper_body_metrics"] = upper

    overlay_path = run_dir / "video" / "skeleton_overlay.mp4"
    create_skeleton_overlay(
        saved_smpl,
        frames,
        overlay_path,
        fps=source_fps,
    )

    validate_report_images(report, run_dir)

    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2)
    )
    (run_dir / "biomechanics.json").write_text(
        json.dumps(bio, indent=2)
    )

    for old in list(run.metrics) + list(run.artifacts):
        db.session.delete(old)

    unique_penalties = set()

    for cycle in report["cycles"]:
        for position, pdata in cycle["positions"].items():
            if pdata.get("missing"):
                continue

            frame_data = pdata.get("frames", {}).get(
                str(pdata.get("event_frame")),
                {},
            )
            metrics = list(
                frame_data.get("metrics", {}).values()
            )
            metrics.extend(pdata.get("upper_body_metrics", []))

            for result in metrics:
                if result.get("penalty"):
                    unique_penalties.add(result["metric_id"])

                db.session.add(MetricResult(
                    analysis_run_id=run.id,
                    gait_cycle=int(cycle["gait_cycle"]),
                    position=position,
                    frame=int(result["frame"]),
                    metric_id=result["metric_id"],
                    metric_label=result["metric_label"],
                    algorithm_penalty=bool(
                        result.get("penalty")
                    ),
                    value_text=result.get("value_text"),
                    threshold_text=result.get("threshold_text"),
                    reasoning=result.get("reasoning"),
                    annotated_image_path=result.get(
                        "annotated_image"
                    ),
                ))

    add_artifact(run, run_dir, "input_video", video, run.original_filename)
    add_artifact(run, run_dir, "frames", frames)
    add_artifact(
        run,
        run_dir,
        "lowerbody_annotated_frames",
        run_dir / "_smas_annotated_frames",
    )
    add_artifact(
        run,
        run_dir,
        "upperbody_annotated_frames",
        annotation_dir,
    )
    add_artifact(run, run_dir, "skeleton_overlay_video", overlay_path)
    add_artifact(run, run_dir, "smpl_json", saved_smpl)
    add_artifact(run, run_dir, "lowerbody_json", saved_lower)
    add_artifact(run, run_dir, "report_json", run_dir / "report.json")
    add_artifact(
        run,
        run_dir,
        "biomechanics_json",
        run_dir / "biomechanics.json",
    )

    for kind, path in [
        (
            "rollout",
            config.ROLLOUT_ROOT / key / "rollout.pkl",
        ),
        (
            "mhr_keypoints",
            config.ROLLOUT_ROOT / key / "mhr_kpts.pkl",
        ),
        (
            "smpl_sequence",
            config.SAM3D_DIR
            / "outputs"
            / "smpl_sequences"
            / key
            / "smpl_sequence.pkl",
        ),
    ]:
        saved_copy = run_dir / "saved_artifacts" / kind / path.name
        add_artifact(run, run_dir, kind, path if path.exists() else saved_copy)

    # One point per unique failed measurement, regardless of cycles.
    run.algorithm_score = len(unique_penalties)


def execute(app, run_id):
    with app.app_context():
        run = db.session.get(AnalysisRun, run_id)
        key = pipeline_key(run)
        run_dir = config.APP_DATA_ROOT / run.public_id

        try:
            run.status = "running"
            run.stage = "Preparing video for analysis"
            db.session.commit()

            video = run_dir / "input" / "original_video.mp4"

            dataset_dir = config.SPRINT_VIDEOS_DIR
            dataset_dir.mkdir(parents=True, exist_ok=True)
            pipeline_video = dataset_dir / f"{key}.mp4"
            shutil.copy2(video, pipeline_video)

            source_fps = probe_video_fps(video)

            frames = run_dir / "frames"
            if frames.exists():
                shutil.rmtree(frames)
            frames.mkdir(parents=True, exist_ok=True)

            cmd(
                run_dir,
                "Preparing video frames",
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video),
                    "-vf",
                    f"fps={source_fps:.6f}",
                    "-q:v",
                    "2",
                    str(frames / "%06d.jpg"),
                ],
            )

            env = {
                "SAM3D_DIR": str(config.SAM3D_DIR),
                "MHR_DIR": str(config.MHR_DIR),
                "SMPL_MODEL": str(config.SMPL_MODEL),
                "DATASET_DIR": str(dataset_dir),
                "ROLLOUT_ROOT": str(config.ROLLOUT_ROOT),
                "CONDA_EXE": str(config.CONDA_EXE),
                "CUDA_VISIBLE_DEVICES": config.GPU_INDEX,
                "PYOPENGL_PLATFORM": "egl",
                "SKIP_EXISTING": "1",
                "PROCESSING_FPS": f"{source_fps:.6f}",
                "INPUT_VIDEO": str(video),
                "FRAME_DIR": str(frames),
            }

            run.stage = "Estimating the athlete mesh"
            db.session.commit()
            cmd(
                run_dir,
                "SAM3D and conversion",
                ["bash", "scripts/run_sam3d_mhr_smpl.sh", key],
                {**env, "DEVICE": "cpu"},
            )

            run.stage = "Optimizing the athlete motion"
            db.session.commit()
            cmd(
                run_dir,
                "GPU rollout optimization",
                ["bash", "scripts/run_rollout_all_athletes.sh", key],
                {**env, "DEVICE": "cuda"},
            )

            run.stage = "Building detailed lower-body landmarks"
            db.session.commit()
            cmd(
                run_dir,
                "CPU MHR skeleton fitting",
                ["bash", "scripts/run_mhr_kpts.sh", key],
                {**env, "DEVICE": "cpu"},
            )

            run.stage = "Creating the analysis report"
            db.session.commit()
            cmd(
                run_dir,
                "Exporting JSON datasets",
                [
                    str(config.CONDA_EXE),
                    "run",
                    "--no-capture-output",
                    "-n",
                    config.CONDA_ENV,
                    "python",
                    "export/export_lowerbody.py",
                    "--athlete",
                    key,
                ],
                env,
            )

            smpl = config.SMPL_JSON_DIR / f"{key}.json"
            lower = config.LOWERBODY_JSON_DIR / f"{key}.json"
            if not smpl.exists() or not lower.exists():
                raise FileNotFoundError(
                    "Expected JSON files were not created."
                )

            datasets = run_dir / "datasets"
            datasets.mkdir(exist_ok=True)
            saved_smpl = datasets / "smpl.json"
            saved_lower = datasets / "lowerbody.json"
            shutil.copy2(smpl, saved_smpl)
            shutil.copy2(lower, saved_lower)

            analyze_saved_run(run, run_dir, source_fps, key)

            run.status = "complete"
            run.stage = "Analysis complete"
            run.completed_at = datetime.utcnow()
            db.session.commit()

        except Exception as exc:
            (run_dir / "traceback.txt").write_text(
                traceback.format_exc()
            )
            run.status = "failed"
            run.stage = "Analysis failed"
            run.error_message = str(exc)
            db.session.commit()
