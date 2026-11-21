"""Paired title and image representation shared by training and serving."""
import io
import numpy as np
from PIL import Image


def image_features(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255
    blocks = rgb.reshape(4, 16, 4, 16, 3)
    features = [blocks.mean(axis=(1, 3)).ravel(), blocks.std(axis=(1, 3)).ravel()]
    features.extend(np.histogram(rgb[:, :, channel], bins=16, range=(0, 1))[0] / 4096 for channel in range(3))
    return np.concatenate(features).astype(np.float32)


def paired_features(model, titles, images):
    words = model["text"].transform(titles)
    return np.hstack((words, np.asarray(images)))
