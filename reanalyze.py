"""
Re-run the analysis stage for saved runs without the GPU pipeline.

Use this after changing the event detector, a metric, or a threshold. It
reuses each run's saved frames and datasets/*.json and regenerates
report.json, biomechanics.json, annotations, the overlay video, and the
MetricResult rows.

Analyst overrides on metric rows are discarded, because they refer to the
old results. Notes and the final-score override are kept.

    python reanalyze.py <public_id> [<public_id> ...]
    python reanalyze.py --all
"""
from __future__ import annotations

import json
import sys

from flask import Flask

import config
from models import db, AnalysisRun
from pipeline import analyze_saved_run, pipeline_key, probe_video_fps

# A standalone app, not app.py: importing app.py would mark in-progress
# server jobs as interrupted.
app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{config.DATABASE_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
db.init_app(app)


def source_fps(run_dir):
    report_path = run_dir / "report.json"
    if report_path.exists():
        fps = json.loads(report_path.read_text()).get("alignment", {}).get("source_fps")
        if fps:
            return float(fps)
    return probe_video_fps(run_dir / "input" / "original_video.mp4")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    with app.app_context():
        if argv == ["--all"]:
            runs = AnalysisRun.query.filter_by(status="complete").all()
        else:
            runs = []
            for public_id in argv:
                run = AnalysisRun.query.filter_by(public_id=public_id).first()
                if run is None:
                    print(f"No run with id {public_id}")
                    return 1
                runs.append(run)
        for run in runs:
            run_dir = config.APP_DATA_ROOT / run.public_id
            print(f"Re-analyzing {run.player.athlete_name} ({run.public_id})...", flush=True)
            try:
                analyze_saved_run(run, run_dir, source_fps(run_dir), pipeline_key(run))
                db.session.commit()
                print(f"  done, algorithm score {run.algorithm_score}")
            except Exception as exc:
                db.session.rollback()
                print(f"  failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
