"""
Face detection & preprocessing.

Locates a face in a frame, crops and aligns it, and resizes it to the
dimensions expected by the embedding model in face_recognition_model.py.
"""

import cv2
import numpy as np

from . import config


class FaceDetector:
    def __init__(self, cascade_name: str = config.FACE_DETECTOR_MODEL):
        cascade_path = cv2.data.haarcascades + cascade_name
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise IOError(f"Could not load face cascade from {cascade_path}")

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes (x, y, w, h) for every face found in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalize lighting before detection
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [tuple(box) for box in faces]

    def preprocess(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Crop the face, resize to the embedding model's expected input, and normalize."""
        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        face = cv2.resize(face, config.FACE_IMAGE_SIZE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        return face

    def get_largest_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Convenience method: detect faces and return the preprocessed largest one, if any."""
        faces = self.detect_faces(frame)
        if not faces:
            return None
        largest = max(faces, key=lambda box: box[2] * box[3])
        return self.preprocess(frame, largest)


if __name__ == "__main__":
    # Quick manual test: run `python -m src.face_detection path/to/image.jpg`
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.face_detection <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Could not read image.")
        sys.exit(1)

    detector = FaceDetector()
    result = detector.get_largest_face(image)
    if result is None:
        print("No face detected.")
    else:
        print(f"Face detected and preprocessed to shape {result.shape}")
