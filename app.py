from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from models import db, Player, AnalysisRun, MetricResult, Note
from pipeline import submit, recover_interrupted_runs
from scoring_adapter import calculate_position_frame
from biomechanics import analyze, metrics_for_position, draw_upper_annotation

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{config.DATABASE_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
)
db.init_app(app)

with app.app_context():
    db.create_all()
    recover_interrupted_runs()


def required(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        if not session.get("auth"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrap


def safe_name(value):
    value = re.sub(r"\s+", "_", value.strip())
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Enter a valid athlete name.")
    return value


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not config.LOGIN_PASSWORD:
            flash("Login is disabled: set SMAS_LOGIN_PASSWORD on the server.")
            return render_template("login.html")
        username_ok = hmac.compare_digest(
            request.form.get("username", "").encode(), config.LOGIN_USERNAME.encode()
        )
        password_ok = hmac.compare_digest(
            request.form.get("password", "").encode(), config.LOGIN_PASSWORD.encode()
        )
        if username_ok and password_ok:
            session["auth"] = True
            return redirect(url_for("dashboard"))
        flash("Incorrect username or password.")
    return render_template("login.html")


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@required
def dashboard():
    runs = AnalysisRun.query.order_by(AnalysisRun.created_at.desc()).all()
    return render_template("dashboard.html", runs=runs)


@app.post("/upload")
@required
def upload():
    try:
        athlete = safe_name(request.form.get("athlete", ""))
        recording_date = datetime.strptime(
            request.form.get("recording_date", ""),
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("dashboard"))

    video = request.files.get("video")
    filename = secure_filename(video.filename if video else "")
    if not video or Path(filename).suffix.lower() != ".mp4":
        flash("Choose an MP4 video.")
        return redirect(url_for("dashboard"))

    temporary_dir = config.APP_DATA_ROOT / "_uploads"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / f"{uuid.uuid4().hex}_{filename}"
    video.save(temporary_path)
    digest = hash_file(temporary_path)

    player = Player.query.filter_by(athlete_name=athlete).first()
    if player is None:
        player = Player(athlete_name=athlete)
        db.session.add(player)
        db.session.flush()

    # Failed runs do not block a retry.
    duplicate = (
        AnalysisRun.query
        .filter(
            AnalysisRun.player_id == player.id,
            AnalysisRun.recording_date == recording_date,
            AnalysisRun.video_sha256 == digest,
            AnalysisRun.status.in_(["queued", "running", "complete"]),
        )
        .order_by(AnalysisRun.created_at.desc())
        .first()
    )

    if duplicate:
        temporary_path.unlink(missing_ok=True)
        return render_template(
            "duplicate.html",
            run=duplicate,
            athlete=athlete,
            recording_date=recording_date,
        )

    public_id = uuid.uuid4().hex
    run_dir = config.APP_DATA_ROOT / public_id
    (run_dir / "input").mkdir(parents=True)
    temporary_path.replace(run_dir / "input" / "original_video.mp4")

    run = AnalysisRun(
        public_id=public_id,
        player_id=player.id,
        recording_date=recording_date,
        original_filename=filename,
        video_sha256=digest,
        status="queued",
        stage="Waiting to start",
    )
    db.session.add(run)
    db.session.commit()
    submit(app, run.id)
    return redirect(url_for("status", public_id=public_id))


@app.get("/runs/<public_id>/status")
@required
def status(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    return render_template("status.html", run=run)


@app.get("/api/runs/<public_id>")
@required
def api_status(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    return jsonify(
        status=run.status,
        stage=run.stage,
        error_message=run.error_message,
        report_url=(
            url_for("report", public_id=public_id)
            if run.status == "complete"
            else None
        ),
    )


@app.get("/runs/<public_id>")
@required
def report(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    run_dir = config.APP_DATA_ROOT / public_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        abort(404)

    payload = json.loads(report_path.read_text())
    notes = {
        f"{n.gait_cycle or ''}|{n.position or ''}|{n.metric_id or ''}":
        n.note_text
        for n in run.notes
    }
    overrides = {
        f"{m.gait_cycle}|{m.position}|{m.frame}|{m.metric_id}":
        (m.override_value or "")
        for m in run.metrics
    }

    frame_count = len(list((run_dir / "frames").glob("*.jpg")))
    return render_template(
        "report.html",
        run=run,
        payload=payload,
        notes_map=notes,
        metric_overrides=overrides,
        available_frames=list(range(frame_count)),
    )


@app.post("/api/runs/<public_id>/metric-override")
@required
def metric_override(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    data = request.get_json(force=True)
    metric_row = MetricResult.query.filter_by(
        analysis_run_id=run.id,
        gait_cycle=int(data["gait_cycle"]),
        position=data["position"],
        frame=int(data["frame"]),
        metric_id=data["metric_id"],
    ).first_or_404()

    value = data.get("override_value") or None
    if value not in {None, "pass", "fail"}:
        abort(400)

    metric_row.override_value = value
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/runs/<public_id>/metric-group-override")
@required
def metric_group_override(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    data = request.get_json(force=True)
    value = data.get("override_value") or None
    if value not in {None, "pass", "fail"}:
        abort(400)

    rows = MetricResult.query.filter_by(
        analysis_run_id=run.id,
        metric_id=data["metric_id"],
    ).all()
    for row in rows:
        row.override_value = value
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/runs/<public_id>/final-score")
@required
def final_score(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    value = request.get_json(force=True).get("value")
    run.final_override_score = None if value in ("", None) else int(value)
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/runs/<public_id>/note")
@required
def note(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    data = request.get_json(force=True)

    query = Note.query.filter_by(
        analysis_run_id=run.id,
        gait_cycle=data.get("gait_cycle"),
        position=data.get("position") or None,
        metric_id=data.get("metric_id") or None,
    )
    note_row = query.first()
    if note_row is None:
        note_row = Note(
            analysis_run_id=run.id,
            gait_cycle=data.get("gait_cycle"),
            position=data.get("position") or None,
            metric_id=data.get("metric_id") or None,
        )

    note_row.note_text = (data.get("note_text") or "").strip()
    db.session.add(note_row)
    db.session.commit()
    return jsonify(ok=True)


@app.get("/runs/<public_id>/files/<path:filename>")
@required
def run_file(public_id, filename):
    return send_from_directory(config.APP_DATA_ROOT / public_id, filename)



@app.post("/runs/<public_id>/recalculate-position")
@required
def recalculate_position(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    run_dir = config.APP_DATA_ROOT / public_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        abort(404)

    cycle_id = int(request.form["gait_cycle"])
    position = request.form["position"]
    frame = int(request.form["frame"])

    lower_json = run_dir / "datasets" / "lowerbody.json"
    smpl_json = run_dir / "datasets" / "smpl.json"
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("*.jpg"))

    if frame < 0 or frame >= len(frame_files):
        flash(f"Frame {frame} is outside the available image range 0–{max(len(frame_files)-1, 0)}.")
        return redirect(url_for("report", public_id=public_id))

    try:
        lower_frame = calculate_position_frame(
            lower_json, frames_dir, run_dir, cycle_id, position, frame
        )
        bio = analyze(smpl_json, lower_json)
        if frame >= len(bio["frames"]):
            raise IndexError(
                f"Frame {frame} is outside the biomechanics JSON range 0–{len(bio['frames'])-1}."
            )

        upper = metrics_for_position(bio["frames"][frame], cycle_id, position)
        source = frame_files[frame]
        annotation_dir = run_dir / "annotations" / "upper_body"
        for result in upper:
            destination = annotation_dir / (
                f"cycle_{cycle_id:02d}_{position}_frame_{frame:05d}_{result['metric_id']}.jpg"
            )
            draw_upper_annotation(
                source, destination, result, bio["frames"][frame], bio["joint_names"]
            )
            result["annotated_image"] = str(destination.relative_to(run_dir))

        payload = json.loads(report_path.read_text())
        target_cycle = next(
            (c for c in payload.get("cycles", []) if int(c.get("gait_cycle")) == cycle_id),
            None,
        )
        if target_cycle is None:
            abort(404)
        pdata = target_cycle.setdefault("positions", {}).setdefault(position, {})
        pdata.update({
            "position": position,
            "label": payload.get("position_labels", {}).get(position, position.title()),
            "event_frame": frame,
            "frame_source": "analyst_selected",
            "missing": False,
            "message": None,
            "upper_body_metrics": upper,
            "analyst_status": "analyst_selected",
        })
        pdata.setdefault("frames", {})[str(frame)] = lower_frame
        report_path.write_text(json.dumps(payload, indent=2))

        MetricResult.query.filter_by(
            analysis_run_id=run.id, gait_cycle=cycle_id, position=position
        ).delete(synchronize_session=False)

        results = list(lower_frame.get("metrics", {}).values()) + list(upper)
        for result in results:
            db.session.add(MetricResult(
                analysis_run_id=run.id,
                gait_cycle=cycle_id,
                position=position,
                frame=frame,
                metric_id=result["metric_id"],
                metric_label=result["metric_label"],
                algorithm_penalty=bool(result.get("penalty")),
                value_text=result.get("value_text"),
                threshold_text=result.get("threshold_text"),
                reasoning=result.get("reasoning"),
                annotated_image_path=result.get("annotated_image"),
            ))

        db.session.flush()
        run.algorithm_score = len({
            row.metric_id for row in run.metrics if row.algorithm_penalty
        })
        db.session.commit()
        flash(f"{position.replace('_', ' ').title()} recalculated using frame {frame}.")
    except Exception as exc:
        db.session.rollback()
        flash(f"Could not recalculate the position: {exc}")

    return redirect(url_for("report", public_id=public_id))


@app.post("/runs/<public_id>/missing-position-status")
@required
def missing_position_status(public_id):
    run = AnalysisRun.query.filter_by(public_id=public_id).first_or_404()
    run_dir = config.APP_DATA_ROOT / public_id
    report_path = run_dir / "report.json"
    payload = json.loads(report_path.read_text())
    cycle_id = int(request.form["gait_cycle"])
    position = request.form["position"]
    status_value = request.form["status"]
    if status_value not in {"prediction_missing", "not_present_in_video", "unable_to_determine"}:
        abort(400)
    cycle = next(c for c in payload["cycles"] if int(c["gait_cycle"]) == cycle_id)
    pdata = cycle["positions"][position]
    pdata["analyst_status"] = status_value
    report_path.write_text(json.dumps(payload, indent=2))
    flash("Missing-position review status saved.")
    return redirect(url_for("report", public_id=public_id))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
