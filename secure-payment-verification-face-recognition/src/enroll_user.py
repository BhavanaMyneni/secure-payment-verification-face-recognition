"""
Enroll a user: compute their reference face embedding from a folder of their
photos and save it as JSON (matching the shape expected by app/models.py's
UserProfile.face_embedding field).

Run:
    python -m src.enroll_user --username bhavana --images data/dataset/bhavana --out enrollments/bhavana.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .face_detection import FaceDetector
from .face_recognition_model import FaceEmbedder


def enroll(username: str, image_dir: str, output_path: str):
    detector = FaceDetector()
    embedder = FaceEmbedder()

    embeddings = []
    for image_path in sorted(Path(image_dir).glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        face = detector.get_largest_face(frame)
        if face is None:
            print(f"No face found in {image_path}, skipping.")
            continue
        embeddings.append(embedder.embed(face))

    if not embeddings:
        raise ValueError(f"No usable face images found in {image_dir}")

    # Average the embeddings across all enrollment photos for a more robust reference vector.
    reference_embedding = np.mean(embeddings, axis=0)
    reference_embedding = reference_embedding / np.linalg.norm(reference_embedding)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"username": username, "face_embedding": reference_embedding.tolist()}, f, indent=2)

    print(f"Enrolled '{username}' from {len(embeddings)} photo(s) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a user's reference face embedding.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--images", required=True, help="Folder of the user's enrollment photos")
    parser.add_argument("--out", required=True, help="Where to write the enrollment JSON")
    args = parser.parse_args()

    enroll(args.username, args.images, args.out)
