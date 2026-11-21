"""Frequency-band statistics for a scratch ASVspoof gradient-boosted detector."""
import numpy as np


def spectral_features(spectrograms):
    # N x 1 x 128 frequency bins x 192 time bins, identical to the CNN input.
    x = np.asarray(spectrograms, dtype=np.float32)[:, 0]
    return np.concatenate((x.mean(axis=2), x.std(axis=2), np.abs(np.diff(x, axis=2)).mean(axis=2),
                           x.max(axis=2), x.reshape(-1, 32, 4, 4, 48).mean(axis=(2,4)).reshape(len(x), -1)), axis=1)
