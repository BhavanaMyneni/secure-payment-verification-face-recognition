"""
Central configuration for the face-recognition + OTP verification pipeline.
Values are read from environment variables so secrets never live in source control.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Face detection
FACE_DETECTOR_MODEL = os.getenv("FACE_DETECTOR_MODEL", "haarcascade_frontalface_default.xml")
FACE_IMAGE_SIZE = (160, 160)  # width, height expected by the embedding model

# Face recognition
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "face_embedding_model.h5"
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.75"))  # cosine similarity cutoff

# OTP
OTP_SECRET_ENV_VAR = "OTP_SECRET"          # each user's TOTP secret should be stored per-user, not hardcoded
OTP_VALIDITY_SECONDS = int(os.getenv("OTP_VALIDITY_SECONDS", "30"))

# Database (used by the Django app)
DB_NAME = os.getenv("DB_NAME", "payment_verification")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
