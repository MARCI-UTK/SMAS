from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
db=SQLAlchemy()
class Player(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    athlete_name=db.Column(db.String(200),nullable=False,unique=True,index=True)
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False)
    runs=db.relationship('AnalysisRun',back_populates='player',cascade='all, delete-orphan')
class AnalysisRun(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    public_id=db.Column(db.String(40),nullable=False,unique=True,index=True)
    player_id=db.Column(db.Integer,db.ForeignKey('player.id'),nullable=False,index=True)
    recording_date=db.Column(db.Date,nullable=False,index=True)
    original_filename=db.Column(db.String(500),nullable=False)
    video_sha256=db.Column(db.String(64),nullable=False,index=True)
    status=db.Column(db.String(40),nullable=False,default='queued')
    stage=db.Column(db.String(200),nullable=False,default='Queued')
    error_message=db.Column(db.Text)
    algorithm_score=db.Column(db.Integer)
    final_override_score=db.Column(db.Integer)
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False)
    completed_at=db.Column(db.DateTime)
    player=db.relationship('Player',back_populates='runs')
    artifacts=db.relationship('Artifact',back_populates='run',cascade='all, delete-orphan')
    metrics=db.relationship('MetricResult',back_populates='run',cascade='all, delete-orphan')
    notes=db.relationship('Note',back_populates='run',cascade='all, delete-orphan')
    @property
    def reviewed_score(self):
        """Score after analyst overrides: one point per metric_id with any active penalty.
        Matches updateScoreAndTable() in report.html. Review-only metrics always have
        algorithm_penalty=False, so they count only when an analyst sets them to 'fail'."""
        active={m.metric_id for m in self.metrics if m.override_value=='fail' or (m.override_value is None and m.algorithm_penalty)}
        return len(active)
    @property
    def display_score(self):
        if self.status!='complete':
            return None
        return self.final_override_score if self.final_override_score is not None else self.reviewed_score
class Artifact(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    analysis_run_id=db.Column(db.Integer,db.ForeignKey('analysis_run.id'),nullable=False)
    artifact_type=db.Column(db.String(100),nullable=False,index=True)
    relative_path=db.Column(db.String(1000),nullable=False)
    original_filename=db.Column(db.String(500))
    file_size_bytes=db.Column(db.Integer)
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False)
    run=db.relationship('AnalysisRun',back_populates='artifacts')
class MetricResult(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    analysis_run_id=db.Column(db.Integer,db.ForeignKey('analysis_run.id'),nullable=False)
    gait_cycle=db.Column(db.Integer,nullable=False)
    position=db.Column(db.String(50),nullable=False)
    frame=db.Column(db.Integer,nullable=False)
    metric_id=db.Column(db.String(150),nullable=False)
    metric_label=db.Column(db.String(300),nullable=False)
    algorithm_penalty=db.Column(db.Boolean,nullable=False)
    override_value=db.Column(db.String(20))
    value_text=db.Column(db.Text)
    threshold_text=db.Column(db.Text)
    reasoning=db.Column(db.Text)
    annotated_image_path=db.Column(db.String(1000))
    run=db.relationship('AnalysisRun',back_populates='metrics')
class Note(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    analysis_run_id=db.Column(db.Integer,db.ForeignKey('analysis_run.id'),nullable=False)
    gait_cycle=db.Column(db.Integer)
    position=db.Column(db.String(50))
    metric_id=db.Column(db.String(150))
    note_text=db.Column(db.Text,nullable=False,default='')
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False)
    updated_at=db.Column(db.DateTime,default=datetime.utcnow,onupdate=datetime.utcnow,nullable=False)
    run=db.relationship('AnalysisRun',back_populates='notes')
