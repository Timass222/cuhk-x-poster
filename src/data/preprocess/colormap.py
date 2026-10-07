"""Invert the Depth_Color rainbow palette back to scalar depth.

The colorizer maps depth to a hue ramp (red 0 deg -> blue 240 deg, i.e.
hue in [0, 2/3] after normalisation), with saturation/value ~1. Black
pixels carry no depth. Inversion is therefore analytic -- no LUT table
needed:

    scalar = hue / (2/3)   in [0, 1],   mask = pixel is not black

Whether 0 means near or far does not matter for a CNN, but the scalar is
metric-monotonic, which the raw RGB image is not (the palette wraps
through 6 RGB segments, so nets fed raw RGB must first unlearn the wrap).
"""

from __future__ import annotations

import numpy as np


def rgb_to_depth(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint8 -> (depth [0,1] float32, valid mask bool)."""
    rgb = img[..., :3].astype(np.float32) / 255.0
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    delta = mx - mn
    valid = mx > 0.05
    hue = np.zeros_like(mx)
    safe = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rc = np.where(safe, (mx - r) / np.where(safe, delta, 1), 0)
    gc = np.where(safe, (mx - g) / np.where(safe, delta, 1), 0)
    bc = np.where(safe, (mx - b) / np.where(safe, delta, 1), 0)
    hue = np.where(r == mx, bc - gc,
                   np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = (hue / 6.0) % 1.0
    depth = np.clip(hue / (2.0 / 3.0), 0.0, 1.0).astype(np.float32)
    return np.where(valid, depth, 0.0).astype(np.float32), valid
