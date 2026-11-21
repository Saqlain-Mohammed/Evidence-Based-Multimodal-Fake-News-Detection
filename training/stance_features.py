"""Sentence-pair features shared by scratch training and evidence comparison."""
import re

from sklearn.feature_extraction import FeatureHasher

LABELS = ("entailment", "neutral", "contradiction")
WORDS = re.compile(r"[\w']+")
HASHER = FeatureHasher(n_features=2**17, input_type="dict")


def pair_features(premise, hypothesis):
    left = WORDS.findall(premise.lower())[:160]
    right = WORDS.findall(hypothesis.lower())[:160]
    common = set(left) & set(right)
    features = {"coverage": len(common) / max(1, len(set(right))),
                "length difference": (len(left) - len(right)) / max(1, len(left) + len(right))}
    for prefix, words in (("common", common), ("added", set(right) - common),
                          ("removed", set(left) - common), ("hypothesis", right)):
        features.update({f"{prefix} {word}": 1 for word in words})
    features.update({f"hypothesis {a} {b}": 1 for a, b in zip(right, right[1:])})
    return features


def encode(pairs):
    return HASHER.transform(pair_features(p, h) for p, h in pairs)
