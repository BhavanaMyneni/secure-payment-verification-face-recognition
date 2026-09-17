# Secure Payment Verification System — Face Recognition

A biometric authentication pipeline for online payments that combines face recognition with OTP-based two-factor verification.

## How it works
1. **Face detection & preprocessing** (`src/face_detection.py`) — locates and aligns a face in the input frame using OpenCV.
2. **Face recognition** (`src/face_recognition_model.py`) — generates a face embedding and compares it against the enrolled user's stored embedding.
3. **OTP verification** (`src/otp_service.py`) — on a successful face match, generates and validates a time-based one-time password as the second factor.
4. **Backend & storage** (`app/`) — Django app exposing enrollment and verification endpoints, backed by MySQL.

## Project structure
```
secure-payment-verification-face-recognition/
├── src/
│   ├── face_detection.py        # OpenCV face detection + alignment
│   ├── face_recognition_model.py # Embedding model + matching logic
│   ├── otp_service.py           # OTP generation & verification
│   └── config.py                # Central configuration
├── app/                          # Django backend (enrollment + verification API)
├── models/                       # Saved model weights (not committed — see .gitignore)
├── data/                         # Local sample/test images (not committed)
├── tests/                        # Unit tests
├── requirements.txt
└── .gitignore
```

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training the embedding model
The recognition model isn't pretrained — you train it yourself on a small face dataset:

1. Collect images: `data/dataset/<person_name>/*.jpg` (5+ people, 10-20 photos each works for a demo).
2. Train: `python -m src.train_embedding_model --data_dir data/dataset --epochs 20`
   This fine-tunes a MobileNetV2-backed embedding head with triplet loss and saves
   `models/face_embedding_model.h5`.
3. Enroll a user: `python -m src.enroll_user --username <name> --images data/dataset/<name> --out enrollments/<name>.json`
   This computes and saves their reference embedding for later verification.

## Status
🚧 Actively being rebuilt piece by piece. Current progress:
- [x] Project scaffold
- [x] Face detection & preprocessing
- [x] Embedding model training script (triplet loss, transfer learning)
- [x] User enrollment script
- [ ] Trained model + enrolled test users (run training yourself — see above)
- [x] OTP verification layer
- [x] Django backend + MySQL integration (endpoints wired, DB not yet provisioned)
- [ ] Tests for embedding/enrollment/OTP modules (only face detection covered so far)

## Tech stack
Python, OpenCV, TensorFlow/Keras, Django, MySQL, pyotp (OTP), SSL (transport security)
