"""Skeleton JSON sequence -> (T, 17, 3) tensor [x, y, score] per clip.

Frames are the per-frame COCO-17 predictions from the RGB camera
(predictions/Color_<ts>_<idx>.json). Coordinates arrive root-centred
(joint 0 ~ (0,0)) and roughly unit-scaled. Per frame there can be 0, 1 or
2+ detected people: we keep the person with the highest mean keypoint
score; a frame with no detection becomes NaN (interpolated by the dataset
loader, masked in the model).
"""

from __future__ import annotations

import json

import numpy as np


def parse_frame(payload) -> np.ndarray:
    """One JSON document -> (17, 3) or NaN frame."""
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if not isinstance(payload, list) or not payload:
        return np.full((17, 3), np.nan, dtype=np.float32)
    best, best_score = None, -1.0
    for person in payload:
        kp = person.get("keypoints")
        sc = person.get("keypoint_scores")
        if kp is None or len(kp) != 17:
            continue
        mean_sc = float(np.mean(sc)) if sc else 0.0
        if mean_sc > best_score:
            best, best_score = kp, mean_sc
    if best is None:
        return np.full((17, 3), np.nan, dtype=np.float32)
    return np.asarray(best, dtype=np.float32)


def stack_clip(frame_payloads: list) -> np.ndarray:
    """Ordered per-frame payloads -> (T, 17, 3)."""
    if not frame_payloads:
        return np.full((1, 17, 3), np.nan, dtype=np.float32)
    return np.stack([parse_frame(p) for p in frame_payloads])
