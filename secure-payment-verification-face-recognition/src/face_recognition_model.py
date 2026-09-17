"""
Face embedding & matching.

Loads a trained embedding model (e.g. a FaceNet-style CNN saved to
models/face_embedding_model.h5) and compares embeddings using cosine
similarity to decide whether two faces belong to the same person.

NOTE: This module expects a trained model file at config.EMBEDDING_MODEL_PATH.
Training/fine-tuning that model is a separate step — see the project README
"Status" checklist.
"""

from __future__ import annotations

import numpy as np

from . import config

try:
    from tensorflow import keras
except ImportError:  # allows the rest of the pipeline to be imported/tested without TF installed
    keras = None


class FaceEmbedder:
    def __init__(self, model_path=config.EMBEDDING_MODEL_PATH):
        if keras is None:
            raise ImportError("tensorflow is required to load the embedding model. pip install tensorflow")
        self._model = keras.models.load_model(model_path)

    def embed(self, face: np.ndarray) -> np.ndarray:
        """face: preprocessed image array from FaceDetector.preprocess(). Returns a 1D embedding vector."""
        batch = np.expand_dims(face, axis=0)
        embedding = self._model.predict(batch, verbose=0)[0]
        return embedding / np.linalg.norm(embedding)  # L2-normalize for cosine comparison


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def is_match(embedding: np.ndarray, enrolled_embedding: np.ndarray, threshold: float = config.MATCH_THRESHOLD) -> bool:
    """Compare a freshly captured embedding against the user's enrolled embedding."""
    similarity = cosine_similarity(embedding, enrolled_embedding)
    return similarity >= threshold
