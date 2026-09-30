# Changelog

v9 is the current version. The v6 and v8 notes are carried over from the original development folder (`UPDATES_V6.md`, `UPDATES_V8.md`).

## v9 (repository handoff)

**Fixes**
- **Recalculate using this frame works again.** `app.py` was missing the imports for `calculate_position_frame`, `analyze`, `metrics_for_position`, and `draw_upper_annotation`, so every recalculation failed with a `NameError`.
- **Each run now uses its own key in the external pipeline** (`<athlete>_<first 8 chars of run id>`) instead of the bare athlete name. Before, two videos of the same athlete shared `sprint_videos/<athlete>.mp4`, `ROLLOUT_ROOT/<athlete>/`, and the exported JSON, and with `SKIP_EXISTING=1` a later video could reuse an earlier video's reconstruction. The key is saved as `report.json → pipeline_key`.
- **A frame-count mismatch between the SMPL and lower-body JSON no longer fails the run.** `biomechanics.analyze()` now analyzes the overlapping frames. The mismatch appears in the report's alignment warning, and any event frame beyond the SMPL data skips its upper-body metrics with a warning.
- **Runs interrupted by a server restart** are marked failed at startup with an explanation, instead of showing "running" forever. They can be re-uploaded.
- The status page shows error messages as text instead of HTML.

**Security**
- The login password is no longer hard-coded. `SMAS_LOGIN_PASSWORD` must be set, otherwise login is disabled with a message saying so. Credentials are compared in constant time.
- If `SMAS_SECRET_KEY` is not set, a random key is generated once and stored in `instance/secret_key`. This replaces the hard-coded default.

**Changes**
- The SMAS scoring core (`core/smas_scoring_logic.py`) is now versioned in this repository, and `SMAS_CORE_SCRIPT` defaults to it.
- The input-video folder is configurable (`SPRINT_VIDEOS_DIR`) instead of hard-coded in `pipeline.py`.
- The analysis stage is split out of `execute()` into `pipeline.analyze_saved_run()`, which needs only a run's saved frames and JSON.
- New `reanalyze.py` re-scores saved runs without the GPU pipeline, e.g. after a threshold change.
- The dashboard shows each run's score: the final override if set, otherwise the score after analyst overrides (`AnalysisRun.reviewed_score`).
- New `tests/smoke_test.py`: an end-to-end test on synthetic data with no GPU needed.
- Removed the unused `config.PROCESSING_FPS`.

No database schema changes, so existing `smas.sqlite3` files work as they are.

## SMAS Flask v8 updates

- Added Previous, Next, frame dropdown, and **Recalculate using this frame** controls to every position tab.
- Recalculation uses saved JSON and extracted images only; it does not rerun SAM3D or rollout optimization.
- Added missing-prediction review workflow: select a replacement frame, mark the position absent before the video ends, or mark it unable to determine.
- Removed `Source: new_algorithm` from the report.
- Removed the automatic-annotation sentence from the header.
- Updated Toe Off and Touchdown plain-language descriptions.
- Added extracted-image/lower-body-JSON/SMPL-JSON frame-count validation and report warnings.
- Made sprint-direction arrow bright blue and thicker.
- Increased contrast of left/right hip labels in the rotation annotation.

## SMAS Flask v6 updates

- Preserves the v5 Flask application, authentication, upload pipeline, database persistence, notes, overrides, saved runs, videos, overview table, and artifacts.
- Uses the orange, gray, and white styling throughout the report.
- Removes the Calculate interaction. The original event frame and every available annotated measurement image are displayed automatically.
- Constrains images to the report width and a 650-pixel maximum height.
- Adds a review-only lumbar extension / anterior pelvic tilt measurement at Late Swing and Touchdown.
  - Displays forward lean, signed trunk curvature, and trailing-hip position normalized by torso length.
  - Suggests review when signed extension bend is greater than 12 degrees and trailing-hip position is greater than 0.10 torso lengths.
  - Never adds a point automatically. The analyst selects No penalty or Penalty.
- Updates trunk and pelvic rotation.
  - Scores absolute chest-to-pelvis rotation with a provisional 20-degree threshold.
  - Displays chest and pelvis rotation relative to sprint direction as supporting values.
  - Draws shoulder and hip landmarks on the frame and a top-down inset with sprint, pelvis, and chest arrows.

The lumbar and rotation thresholds remain provisional and should be calibrated against coach-labelled examples.
