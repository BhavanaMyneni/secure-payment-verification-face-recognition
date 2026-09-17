"""
Train the face embedding model.

Approach: transfer learning on top of MobileNetV2 (pretrained on ImageNet),
fine-tuned with triplet loss so that embeddings of the same person end up
close together and embeddings of different people end up far apart. This is
the same family of approach FaceNet uses, at a scale that trains reasonably
on a laptop/Colab GPU rather than requiring FaceNet's original training set.

Expected dataset layout:
    data/dataset/
        person_1/
            img_001.jpg
            img_002.jpg
            ...
        person_2/
            img_001.jpg
            ...

For a resume/portfolio project, ~10-20 images per person across 5+ people is
enough to demonstrate the pipeline working end-to-end. Public face datasets
(e.g. LFW) also work if you want more data.

Run:
    python -m src.train_embedding_model --data_dir data/dataset --epochs 20
"""

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config

EMBEDDING_DIM = 128


def build_embedding_model(input_shape=(*config.FACE_IMAGE_SIZE, 3)) -> keras.Model:
    """MobileNetV2 backbone (ImageNet weights) + a dense embedding head, L2-normalized."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False  # freeze the backbone initially; unfreeze later for fine-tuning if desired

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(EMBEDDING_DIM)(x)
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_normalize")(x)

    return keras.Model(inputs, outputs, name="face_embedding_model")


def triplet_loss(margin: float = 0.2):
    """Standard triplet loss: anchor closer to positive than to negative, by at least `margin`."""

    def loss_fn(y_true, y_pred):
        # y_pred is expected to be a stacked [anchor, positive, negative] embedding batch
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=0)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

    return loss_fn


def load_dataset(data_dir: Path) -> dict[str, list[Path]]:
    """Return {person_name: [image paths]} for every subfolder under data_dir."""
    people = {}
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        images = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if len(images) >= 2:  # need at least 2 images per person to form a positive pair
            people[person_dir.name] = images
    if len(people) < 2:
        raise ValueError(
            f"Need at least 2 people with 2+ images each under {data_dir}. Found: {list(people.keys())}"
        )
    return people


def make_triplet_batch(people: dict[str, list[Path]], batch_size: int) -> tf.data.Dataset:
    """Generator that yields (anchor, positive, negative) image triplets."""

    def _load_image(path: Path) -> np.ndarray:
        img = keras.utils.load_img(path, target_size=config.FACE_IMAGE_SIZE)
        return keras.utils.img_to_array(img) / 255.0

    names = list(people.keys())

    def gen():
        while True:
            anchor_name, negative_name = random.sample(names, 2)
            anchor_path, positive_path = random.sample(people[anchor_name], 2)
            negative_path = random.choice(people[negative_name])
            yield (
                _load_image(anchor_path),
                _load_image(positive_path),
                _load_image(negative_path),
            )

    output_signature = (
        tf.TensorSpec(shape=(*config.FACE_IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*config.FACE_IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*config.FACE_IMAGE_SIZE, 3), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size)
    return ds


def train(data_dir: str, epochs: int, batch_size: int, steps_per_epoch: int, output_path: str):
    people = load_dataset(Path(data_dir))
    print(f"Loaded {len(people)} people: {list(people.keys())}")

    model = build_embedding_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = triplet_loss(margin=0.2)

    dataset = make_triplet_batch(people, batch_size)

    for epoch in range(1, epochs + 1):
        epoch_loss = keras.metrics.Mean()
        for step, (anchor, positive, negative) in enumerate(dataset.take(steps_per_epoch)):
            with tf.GradientTape() as tape:
                batch = tf.concat([anchor, positive, negative], axis=0)
                embeddings = model(batch, training=True)
                loss = loss_fn(None, embeddings)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss.update_state(loss)

        print(f"Epoch {epoch}/{epochs} — triplet loss: {epoch_loss.result():.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved trained embedding model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the face embedding model with triplet loss.")
    parser.add_argument("--data_dir", default="data/dataset", help="Folder of person_name/*.jpg subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--output", default=str(config.EMBEDDING_MODEL_PATH))
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.steps_per_epoch, args.output)
