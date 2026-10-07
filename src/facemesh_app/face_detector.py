"""Find the face to track in a whole frame with the short-range face detector of the model bundle.

Decodes the detector the way MediaPipe's face detector graph (Apache-2.0) does: SSD anchors,
weighted non-maximum suppression and a region grown around the strongest face.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .face_region import FaceRegion, ImageSize

INPUT_SIZE = 128
VALUE_RANGE = (-1.0, 1.0)
ANCHOR_STRIDES = (8, 16, 16, 16)
ANCHORS_PER_LAYER = 2
NUM_KEYPOINTS = 6
SCORE_CLIP = 100.0
MIN_SCORE = 0.5
MIN_SUPPRESSION_IOU = 0.3
REGION_SCALE = 1.5
RIGHT_EYE_KEYPOINT = 0
LEFT_EYE_KEYPOINT = 1
OUTPUTS = ("regressors", "classificators")


def _anchor_centers() -> np.ndarray:
    centers = []
    layer = 0
    while layer < len(ANCHOR_STRIDES):
        stride = ANCHOR_STRIDES[layer]
        same_stride = sum(1 for s in ANCHOR_STRIDES[layer:] if s == stride)
        cells = -(-INPUT_SIZE // stride)
        grid = (np.arange(cells) + 0.5) / cells
        ys, xs = np.meshgrid(grid, grid, indexing="ij")
        cell_centers = np.stack([xs.ravel(), ys.ravel()], axis=1)
        centers.append(np.repeat(cell_centers, same_stride * ANCHORS_PER_LAYER, axis=0))
        layer += same_stride
    return np.concatenate(centers)


def _iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    low = np.maximum(boxes[:, :2], box[:2])
    high = np.minimum(boxes[:, 2:], box[2:])
    intersection = np.prod(np.clip(high - low, 0.0, None), axis=1)
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    union = areas + np.prod(box[2:] - box[:2]) - intersection
    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


class FaceDetector:
    """Region around the most confident face in a frame."""

    def __init__(self, net: cv2.dnn.Net):
        self._net = net
        self._anchors = _anchor_centers()

    def find(self, image: np.ndarray) -> Optional[FaceRegion]:
        """Region to read the landmarks from, or None when no face is found."""
        crop = FaceRegion.whole_image().crop(
            image, INPUT_SIZE, VALUE_RANGE, cv2.BORDER_CONSTANT, keep_aspect=True
        )
        self._net.setInput(crop.blob)
        raw_boxes, raw_scores = self._net.forward(list(OUTPUTS))
        logits = np.clip(raw_scores.reshape(-1).astype(np.float64), -SCORE_CLIP, SCORE_CLIP)
        scores = 1.0 / (1.0 + np.exp(-logits))
        candidates = np.flatnonzero(scores >= MIN_SCORE)
        if candidates.size == 0:
            return None

        raw = raw_boxes.reshape(len(self._anchors), -1)[candidates].astype(np.float64)
        anchors = self._anchors[candidates]
        centers = raw[:, 0:2] / INPUT_SIZE + anchors
        sizes = raw[:, 2:4] / INPUT_SIZE
        valid = np.all(sizes >= 0.0, axis=1)
        if not np.any(valid):
            return None
        boxes = np.concatenate([centers - sizes / 2.0, centers + sizes / 2.0], axis=1)[valid]
        keypoints = (
            raw[:, 4 : 4 + 2 * NUM_KEYPOINTS].reshape(-1, NUM_KEYPOINTS, 2) / INPUT_SIZE
            + anchors[:, np.newaxis, :]
        )[valid]
        scores = scores[candidates][valid]
        best = int(np.argmax(scores))
        weights = np.where(_iou(boxes, boxes[best]) > MIN_SUPPRESSION_IOU, scores, 0.0)
        box = weights @ boxes / weights.sum()
        points = np.tensordot(weights, keypoints, axes=1) / weights.sum()

        corners = crop.to_image(
            np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        )
        points = crop.to_image(points)
        image_size: ImageSize = (image.shape[1], image.shape[0])
        region = FaceRegion.from_box(
            corners.min(axis=0),
            corners.max(axis=0),
            points[RIGHT_EYE_KEYPOINT],
            points[LEFT_EYE_KEYPOINT],
            image_size,
        )
        return region.scaled(REGION_SCALE, image_size)
