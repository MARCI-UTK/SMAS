"""
End-to-end smoke test that needs no GPU or external reconstruction code.

It builds a synthetic sprint (frames + lowerbody.json + smpl.json), runs the
real analysis stage (scoring core event detection, lower- and upper-body
metrics, annotations, overlay video), then exercises every web route:
login, dashboard, report, overrides, notes, final score, frame
recalculation, and missing-position review.

The synthetic motion is only realistic enough for the detectors to find
peaks. This test checks that the code runs end to end, not that the
biomechanics are accurate.

    python tests/smoke_test.py          # requires ffmpeg on PATH

Everything is written to a temporary directory; nothing touches instance/.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import uuid
from datetime import date
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="smas_smoke_"))
os.environ["SMAS_APP_DATA_ROOT"] = str(TMP / "player_data")
os.environ["SMAS_DATABASE_PATH"] = str(TMP / "smas.sqlite3")
os.environ["SMAS_LOGIN_USERNAME"] = "tester"
os.environ["SMAS_LOGIN_PASSWORD"] = "smoke-test"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

N_FRAMES = 150
FPS = 60.0
STRIDE = 40          # frames per leg cycle
W, H = 960, 720


def leg(t, phase, hip_x):
    """Return hip, knee, ankle, heel-tip, toe-tip pixel positions for one leg."""
    a = 2 * math.pi * t / STRIDE + phase
    hip = (hip_x, 400.0)
    knee = (hip_x + 70 * math.sin(a), 500 - 45 * max(0.0, math.sin(a)))
    ankle = (knee[0] - 40 * max(0.0, -math.cos(a)) + 20, 600 - 90 * max(0.0, math.sin(a - 0.6)))
    heel = (ankle[0] - 18, ankle[1] + 12)
    toe = (ankle[0] + 38, ankle[1] + 14)
    return hip, knee, ankle, heel, toe


def build_datasets(run_dir: Path):
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True)
    joints = {name: {"pixel_uv": [], "world_m": []} for name in [
        "l_upleg", "r_upleg", "l_lowleg", "r_lowleg", "l_foot", "r_foot",
        "l_heel_tip", "r_heel_tip", "l_toe_tip", "r_toe_tip",
    ]}
    smpl_frames = []
    smpl_names = ["pelvis", "left_hip", "right_hip", "spine2", "neck", "left_shoulder", "right_shoulder"]

    for t in range(N_FRAMES):
        hip_x = 150 + 4.5 * t
        for side, phase in (("l", 0.0), ("r", math.pi)):
            hip, knee, ankle, heel, toe = leg(t, phase, hip_x)
            offset = -6 if side == "l" else 6
            for name, p in (("upleg", hip), ("lowleg", knee), ("foot", ankle), ("heel_tip", heel), ("toe_tip", toe)):
                joints[f"{side}_{name}"]["pixel_uv"].append([p[0], p[1] + offset * 0.1])
            joints[f"{side}_upleg"]["world_m"].append([0.01 * hip_x, 0.9, offset * 0.02])
            for name in ("lowleg", "foot", "heel_tip", "toe_tip"):
                joints[f"{side}_{name}"]["world_m"].append([0.01 * hip_x, 0.5, offset * 0.02])

        lean = 0.15 + 0.05 * math.sin(2 * math.pi * t / STRIDE)
        pelvis = (hip_x, 400.0)
        spine2 = (hip_x + 20, 300.0)
        neck = (hip_x + 38, 200.0)
        smpl_frames.append({
            "kpts_2d": [list(pelvis), [hip_x - 6, 400], [hip_x + 6, 400], list(spine2), list(neck),
                        [hip_x + 25, 210], [hip_x + 45, 212]],
            "spine_up": [math.sin(lean), math.cos(lean), 0.0],
            "chest_fwd": [1.0, 0.0, 0.1 * math.sin(2 * math.pi * t / STRIDE)],
        })

        Image.new("RGB", (W, H), (60, 120, 60)).save(frames_dir / f"{t + 1:06d}.jpg", quality=80)

    lower = {"athlete": "Smoke_Test", "fps": FPS, "n_frames": N_FRAMES,
             "joint_names": list(joints), "joints": joints}
    smpl = {"n_frames": N_FRAMES, "world_up": [0, 1, 0], "sprint_dir": [1, 0, 0],
            "joint_names": smpl_names, "skeleton": [[0, 3], [3, 4], [0, 1], [0, 2], [4, 5], [4, 6]],
            "frames": smpl_frames}
    datasets = run_dir / "datasets"
    datasets.mkdir()
    (datasets / "lowerbody.json").write_text(json.dumps(lower))
    (datasets / "smpl.json").write_text(json.dumps(smpl))


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def main():
    import config
    from app import app
    from models import db, Player, AnalysisRun, MetricResult
    from pipeline import analyze_saved_run, pipeline_key

    public_id = uuid.uuid4().hex
    run_dir = config.APP_DATA_ROOT / public_id
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "input" / "original_video.mp4").write_bytes(b"")
    print(f"Synthetic run in {run_dir}")
    build_datasets(run_dir)

    with app.app_context():
        player = Player(athlete_name="Smoke_Test")
        db.session.add(player)
        db.session.flush()
        run = AnalysisRun(public_id=public_id, player_id=player.id, recording_date=date.today(),
                          original_filename="smoke.mp4", video_sha256="0" * 64, status="running", stage="test")
        db.session.add(run)
        db.session.commit()

        print("Analysis stage")
        analyze_saved_run(run, run_dir, FPS, pipeline_key(run))
        run.status = "complete"
        db.session.commit()

        report = json.loads((run_dir / "report.json").read_text())
        cycles = report["cycles"]
        check(len(cycles) >= 2, f"detected {len(cycles)} gait cycles")
        check(report["alignment"]["warnings"] == [], "frame counts aligned")
        check((run_dir / "video" / "skeleton_overlay.mp4").stat().st_size > 0, "overlay video written")
        n_metrics = MetricResult.query.filter_by(analysis_run_id=run.id).count()
        check(n_metrics > 0, f"{n_metrics} metric rows stored")

        target_cycle, target_position = None, None
        for cycle in cycles:
            for position, pdata in cycle["positions"].items():
                if not pdata.get("missing") and position == "touchdown":
                    target_cycle, target_position = cycle["gait_cycle"], position
                    event_frame = pdata["event_frame"]
                    break
            if target_cycle:
                break
        check(target_cycle is not None, "found a touchdown to recalculate")
        metric_row = MetricResult.query.filter_by(analysis_run_id=run.id).first()

    client = app.test_client()
    print("Web routes")
    r = client.post("/login", data={"username": "tester", "password": "wrong"})
    check(b"Incorrect" in r.data, "wrong password rejected")
    r = client.post("/login", data={"username": "tester", "password": "smoke-test"})
    check(r.status_code == 302, "login succeeds")
    check(client.get("/").status_code == 200, "dashboard renders")
    r = client.get(f"/runs/{public_id}")
    check(r.status_code == 200 and b"Gait Cycle 1" in r.data, "report renders")

    r = client.post(f"/api/runs/{public_id}/metric-override", json={
        "gait_cycle": metric_row.gait_cycle, "position": metric_row.position,
        "frame": metric_row.frame, "metric_id": metric_row.metric_id, "override_value": "fail"})
    check(r.status_code == 200, "metric override saved")
    with app.app_context():
        run = AnalysisRun.query.filter_by(public_id=public_id).first()
        check(run.reviewed_score >= 1, f"reviewed score reflects override ({run.reviewed_score})")
    check(client.post(f"/api/runs/{public_id}/note", json={
        "gait_cycle": None, "position": None, "metric_id": None, "note_text": "smoke"}).status_code == 200, "note saved")
    check(client.post(f"/api/runs/{public_id}/final-score", json={"value": "3"}).status_code == 200, "final score saved")

    new_frame = min(event_frame + 1, N_FRAMES - 1)
    r = client.post(f"/runs/{public_id}/recalculate-position", data={
        "gait_cycle": target_cycle, "position": target_position, "frame": new_frame}, follow_redirects=True)
    check(b"recalculated using frame" in r.data, f"touchdown recalculated at frame {new_frame}")
    report = json.loads((run_dir / "report.json").read_text())
    pdata = next(c for c in report["cycles"] if c["gait_cycle"] == target_cycle)["positions"][target_position]
    check(pdata["event_frame"] == new_frame and pdata["frame_source"] == "analyst_selected", "report.json updated")

    r = client.post(f"/runs/{public_id}/missing-position-status", data={
        "gait_cycle": target_cycle, "position": target_position, "status": "unable_to_determine"},
        follow_redirects=True)
    check(b"review status saved" in r.data, "missing-position status saved")

    print(f"\nSmoke test passed. Output kept in {TMP}")


if __name__ == "__main__":
    main()
