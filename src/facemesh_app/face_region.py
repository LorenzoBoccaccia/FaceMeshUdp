"""Image regions around a face, and the model inputs cut from them.

Follows the region and crop conventions of MediaPipe's face landmarker (Apache-2.0), so the models
of its bundle see the face exactly as they were trained to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

ImageSize = Tuple[int, int]


def _normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


@dataclass(frozen=True)
class FaceCrop:
    """Model input cut from an image region, with the map from its normalized coordinates back to the image."""

    blob: np.ndarray
    matrix: np.ndarray

    def to_image(self, points: np.ndarray) -> np.ndarray:
        """Image-normalized positions of points given in crop-normalized coordinates."""
        return points @ self.matrix[:, :2].T + self.matrix[:, 2]

    @property
    def z_scale(self) -> float:
        """Width of a crop unit in image-normalized units, the scale of landmark depth."""
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))


@dataclass(frozen=True)
class FaceRegion:
    """Rotated rectangle around a face, normalized to the image it lies in."""

    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float = 0.0

    @classmethod
    def whole_image(cls) -> FaceRegion:
        return cls(0.5, 0.5, 1.0, 1.0)

    @classmethod
    def from_box(
        cls,
        box_min: np.ndarray,
        box_max: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        image_size: ImageSize,
    ) -> FaceRegion:
        """The box, turned so that the direction from start to end lies along the region's width."""
        width, height = image_size
        dx = (end[0] - start[0]) * width
        dy = (end[1] - start[1]) * height
        return cls(
            float(box_min[0] + box_max[0]) / 2.0,
            float(box_min[1] + box_max[1]) / 2.0,
            float(box_max[0] - box_min[0]),
            float(box_max[1] - box_min[1]),
            _normalize_radians(-math.atan2(-dy, dx)),
        )

    def scaled(self, scale: float, image_size: ImageSize, square: bool = False) -> FaceRegion:
        """The region grown by scale, first squared on its longer side in pixels when asked."""
        width, height = self.width, self.height
        if square:
            long_side = max(width * image_size[0], height * image_size[1])
            width = long_side / image_size[0]
            height = long_side / image_size[1]
        return FaceRegion(self.x_center, self.y_center, width * scale, height * scale, self.rotation)

    def crop(
        self,
        image: np.ndarray,
        size: int,
        value_range: Tuple[float, float],
        border: int,
        keep_aspect: bool = False,
    ) -> FaceCrop:
        """Square model input of the region, its pixels mapped linearly onto value_range."""
        image_height, image_width = image.shape[:2]
        cx = self.x_center * image_width
        cy = self.y_center * image_height
        width = self.width * image_width
        height = self.height * image_height
        if keep_aspect:
            width = height = max(width, height)

        corners = cv2.boxPoints(((cx, cy), (width, height), math.degrees(self.rotation)))
        target = np.array([[0, size], [0, 0], [size, 0], [size, size]], dtype=np.float32)
        warp = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
        pixels = cv2.warpPerspective(
            image, warp, (size, size), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0
        )
        low, high = value_range
        values = pixels.astype(np.float32) * np.float32((high - low) / 255.0) + np.float32(low)

        cos, sin = math.cos(self.rotation), math.sin(self.rotation)
        matrix = np.array(
            [
                [width * cos, -height * sin, -0.5 * width * cos + 0.5 * height * sin + cx],
                [width * sin, height * cos, -0.5 * height * cos - 0.5 * width * sin + cy],
            ]
        ) / np.array([[image_width], [image_height]])
        return FaceCrop(np.ascontiguousarray(values.transpose(2, 0, 1)[np.newaxis]), matrix)
