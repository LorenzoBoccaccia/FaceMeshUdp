"""Steady the landmarks of a tracked face from frame to frame as MediaPipe's video mode does.

A One Euro filter on every landmark coordinate, in pixels relative to the face size, tuned as in
MediaPipe's face landmarks detector graph (Apache-2.0).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .face_region import ImageSize

MIN_CUTOFF = 0.05
BETA = 80.0
DERIVATE_CUTOFF = 1.0
DEFAULT_FREQUENCY = 30.0
MIN_FACE_SCALE = 1e-6
NS_PER_MS = 1_000_000


def _alpha(cutoff, frequency: float):
    return 1.0 / (1.0 + frequency / (2.0 * math.pi * cutoff))


class LandmarkSmoother:
    """Landmarks of one face, filtered less the faster they move."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget the face, so the next landmarks pass through unfiltered."""
        self._last_time_ns = -1
        self._frequency = DEFAULT_FREQUENCY
        self._raw: Optional[np.ndarray] = None
        self._value: Optional[np.ndarray] = None
        self._velocity: Optional[np.ndarray] = None

    def smooth(self, landmarks: np.ndarray, timestamp_ms: int, image_size: ImageSize) -> np.ndarray:
        """The landmarks after filtering, in the same normalized coordinates."""
        to_pixels = np.array([image_size[0], image_size[1], image_size[0]], dtype=np.float64)
        pixels = landmarks * to_pixels
        extent = pixels[:, :2].max(axis=0) - pixels[:, :2].min(axis=0)
        face_scale = float(extent.sum()) / 2.0
        time_ns = int(timestamp_ms) * NS_PER_MS
        if face_scale < MIN_FACE_SCALE or self._last_time_ns >= time_ns:
            return landmarks
        if self._last_time_ns != 0 and time_ns != 0:
            self._frequency = 1e9 / (time_ns - self._last_time_ns)
        self._last_time_ns = time_ns

        if self._raw is None:
            speed = np.zeros_like(pixels)
        else:
            speed = (pixels - self._raw) / face_scale * self._frequency
        if self._velocity is None:
            self._velocity = speed
        else:
            weight = _alpha(DERIVATE_CUTOFF, self._frequency)
            self._velocity = weight * speed + (1.0 - weight) * self._velocity
        if self._value is None:
            self._value = pixels
        else:
            weight = _alpha(MIN_CUTOFF + BETA * np.abs(self._velocity), self._frequency)
            self._value = weight * pixels + (1.0 - weight) * self._value
        self._raw = pixels
        return self._value / to_pixels
