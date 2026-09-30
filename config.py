from pathlib import Path
import os
import secrets
BASE_DIR=Path(__file__).resolve().parent
CODES_DIR=Path(os.getenv('ATHLETE_CODES_DIR','/data/marci/mvenkat4/Athlete_Mesh_Analysis_Codes'))
SAM3D_DIR=Path(os.getenv('SAM3D_DIR','/data/marci/mvenkat4/sam-3d-body'))
MHR_DIR=Path(os.getenv('MHR_DIR','/data/marci/mvenkat4/MHR'))
SMPL_MODEL=Path(os.getenv('SMPL_MODEL',str(CODES_DIR/'SMPL_N_model_generate_from_npz.pkl')))
ROLLOUT_ROOT=Path(os.getenv('ROLLOUT_ROOT','/data/marci/mvenkat4/outputs_with_moge2/rollout_results'))
SMPL_JSON_DIR=Path(os.getenv('SMPL_JSON_DIR','/data/marci/mvenkat4/sprint_smpl_dataset/data'))
LOWERBODY_JSON_DIR=Path(os.getenv('LOWERBODY_JSON_DIR','/data/marci/mvenkat4/sprint_lowerbody_dataset/data'))
# Input-video folder read by the external reconstruction scripts (passed as DATASET_DIR).
SPRINT_VIDEOS_DIR=Path(os.getenv('SPRINT_VIDEOS_DIR','/data/marci/mvenkat4/sprint_videos'))
# The scoring core is versioned in this repository (core/). Set SMAS_CORE_SCRIPT to use a different copy.
SMAS_CORE_SCRIPT=Path(os.getenv('SMAS_CORE_SCRIPT',str(BASE_DIR/'core'/'smas_scoring_logic.py')))
CONDA_EXE=Path(os.getenv('CONDA_EXE','/data/marci/mvenkat4/miniconda3/bin/conda'))
CONDA_ENV=os.getenv('CONDA_ENV','sam_3d_body')
GPU_INDEX=os.getenv('GPU_INDEX','0')
APP_DATA_ROOT=Path(os.getenv('SMAS_APP_DATA_ROOT',str(BASE_DIR/'instance'/'player_data')))
APP_DATA_ROOT.mkdir(parents=True,exist_ok=True)
DATABASE_PATH=Path(os.getenv('SMAS_DATABASE_PATH',str(BASE_DIR/'instance'/'smas.sqlite3')))
DATABASE_PATH.parent.mkdir(parents=True,exist_ok=True)
def _secret_key():
    """Use SMAS_SECRET_KEY if set; otherwise generate one and keep it in instance/ so sessions survive restarts."""
    if os.getenv('SMAS_SECRET_KEY'):
        return os.environ['SMAS_SECRET_KEY']
    path=DATABASE_PATH.parent/'secret_key'
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()
SECRET_KEY=_secret_key()
LOGIN_USERNAME=os.getenv('SMAS_LOGIN_USERNAME','volfb')
# No default password is committed. Login is disabled until SMAS_LOGIN_PASSWORD is set.
LOGIN_PASSWORD=os.getenv('SMAS_LOGIN_PASSWORD')
MAX_UPLOAD_MB=int(os.getenv('MAX_UPLOAD_MB','1500'))
