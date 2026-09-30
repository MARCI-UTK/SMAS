from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

import config


def load_core():
    path = config.SMAS_CORE_SCRIPT
    if not path.exists():
        raise FileNotFoundError(f"Scoring core not found: {path}")

    spec = importlib.util.spec_from_file_location("smas_core", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _path_from_payload(value):
    if not value:
        return None

    value = str(value)
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path))

    return Path(value)


def _relative_to_run(value, run_dir):
    """
    Convert paths emitted by the standalone scoring core into paths that the
    Flask /runs/<id>/files route can serve.

    The standalone core emits file:// URIs because report_dir is None.
    Flask needs a path relative to the current analysis directory.
    """
    path = _path_from_payload(value)
    if path is None:
        return None

    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return None


def _normalize_image_paths(payload, run_dir):
    for cycle in payload.get("cycles", []):
        for position_data in cycle.get("positions", {}).values():
            for frame_data in position_data.get("frames", {}).values():
                frame_data["raw_image"] = _relative_to_run(
                    frame_data.get("raw_image"),
                    run_dir,
                )

                for metric_data in frame_data.get("metrics", {}).values():
                    metric_data["annotated_image"] = _relative_to_run(
                        metric_data.get("annotated_image"),
                        run_dir,
                    )

    return payload


def build_report(athlete, lower_json, frames_dir):
    core = load_core()

    df, joints, fps, raw = core.load_motion_json(lower_json)

    # ffmpeg extracts frames as six-digit, one-based filenames: 000001.jpg,
    # 000002.jpg, ... . The standalone core normally emits five-digit names,
    # so explicitly align each motion-data frame with its extracted image.
    # Motion frame 0 corresponds to image 000001.jpg.
    df["image_number"] = df["frame"].astype(int) + 1
    df["file_name"] = df["image_number"].map(lambda n: f"{int(n):06d}.jpg")

    df = core.add_feature_columns_for_prediction(df)
    cycles = core.detect_new_algorithm(df)

    rows = []
    for cycle_id, cycle in enumerate(cycles, 1):
        for position in core.POSITION_ORDER:
            if cycle.get(position) is not None:
                rows.append({
                    "gait_cycle": cycle_id,
                    "position": position,
                    "position_norm": position,
                    "frame": int(cycle[position]),
                    "frame_source": "new_algorithm",
                })

    if not rows:
        raise RuntimeError("No gait cycles were detected.")

    events = pd.DataFrame(rows)

    old_dir = core.IMAGE_DIR
    old_offset = core.IMAGE_FRAME_OFFSET

    try:
        # The core creates:
        #   raw frames: <run>/frames
        #   annotations: <run>/_smas_annotated_frames
        core.IMAGE_DIR = str(frames_dir)
        core.IMAGE_FRAME_OFFSET = 1

        payload = core.build_scoring_payload(
            df,
            events,
            {"notes": ["Prediction-only mode."], "sections": []},
            events,
        )
    finally:
        core.IMAGE_DIR = old_dir
        core.IMAGE_FRAME_OFFSET = old_offset

    payload.pop("prediction_performance", None)
    payload["athlete"] = athlete
    payload["fps"] = float(fps)
    payload["predicted_events"] = events.to_dict(orient="records")

    payload = json.loads(json.dumps(
        payload,
        default=lambda x: x.item() if hasattr(x, "item") else str(x),
    ))

    return _normalize_image_paths(payload, Path(frames_dir).parent)


def calculate_position_frame(lower_json, frames_dir, run_dir, cycle_id, position, frame):
    """Recalculate all lower-body metrics and annotations for one selected frame."""
    core = load_core()
    df, _, _, _ = core.load_motion_json(lower_json)
    df["image_number"] = df["frame"].astype(int) + 1
    df["file_name"] = df["image_number"].map(lambda n: f"{int(n):06d}.jpg")
    df = core.add_feature_columns_for_prediction(df)

    frame = int(frame)
    row_matches = df[df["frame"].astype(int) == frame]
    if row_matches.empty:
        raise IndexError(f"Frame {frame} is outside the lower-body JSON range.")

    row = row_matches.iloc[0]
    direction = core.estimate_running_direction(df)
    by_frame = {int(r["frame"]): r for _, r in df.iterrows()}

    old_dir = core.IMAGE_DIR
    old_offset = core.IMAGE_FRAME_OFFSET
    try:
        core.IMAGE_DIR = str(frames_dir)
        core.IMAGE_FRAME_OFFSET = 1
        image_path = core.find_image_for_frame(frame, by_frame, frames_dir)
        raw_image = _relative_to_run(image_path, run_dir)
        metrics = {}
        annotation_dir = Path(run_dir) / "_smas_annotated_frames"
        annotation_dir.mkdir(parents=True, exist_ok=True)

        for metric_id in core.POSITION_METRICS.get(position, []):
            result = core.metric(metric_id, row, int(cycle_id), direction)
            if image_path is not None:
                output_path = annotation_dir / (
                    f"cycle_{int(cycle_id):02d}_{position}_frame_{frame:05d}_{metric_id}.jpg"
                )
                result.annotated_image = core.annotate_metric(
                    result, row, image_path, output_path, direction
                )
            metric_data = core.metric_to_dict(result, None)
            metric_data["annotated_image"] = _relative_to_run(
                metric_data.get("annotated_image"), run_dir
            )
            metrics[metric_id] = metric_data

        return {
            "raw_image": raw_image,
            "selected_heel": core.get_selected_heel_info(row, position, direction),
            "metrics": metrics,
        }
    finally:
        core.IMAGE_DIR = old_dir
        core.IMAGE_FRAME_OFFSET = old_offset
