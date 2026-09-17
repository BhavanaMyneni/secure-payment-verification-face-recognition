"""
Basic tests for FaceDetector.

These focus on the parts that don't require a real trained model or webcam:
that the detector loads, and that preprocessing produces correctly shaped output.
Run with: pytest
"""

import numpy as np
import pytest

from src.face_detection import FaceDetector
from src import config


@pytest.fixture
def detector():
    return FaceDetector()


def test_detector_loads(detector):
    assert detector is not None


def test_preprocess_output_shape(detector):
    # A synthetic "frame" and a fake bounding box, since we're not relying on a real face image here.
    fake_frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50, 50, 100, 100)  # x, y, w, h
    face = detector.preprocess(fake_frame, box)

    assert face.shape == (*config.FACE_IMAGE_SIZE, 3)
    assert face.dtype == np.float32
    assert face.min() >= 0.0 and face.max() <= 1.0


def test_detect_faces_on_blank_frame_returns_list(detector):
    blank_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(blank_frame)
    assert isinstance(faces, list)  # no faces expected on a blank frame, but should not error
