"""Head pose in metric camera space, fitted from the face landmarks to the canonical face of the model bundle.

Reproduces MediaPipe's face geometry pipeline (Apache-2.0): a perspective camera with a 63 degree
vertical field of view, depth recovered by two scale estimates, and a weighted orthogonal
Procrustes fit of the canonical face.
"""

from __future__ import annotations

import math
import struct
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .face_region import ImageSize

VERTICAL_FOV_DEG = 63.0
NEAR_PLANE = 1.0
MESH_LANDMARKS = 468
VERTEX_SIZE = 5
MIN_FACE_EXTENT = 1e-3
FACE_LANDMARK_SOURCES = (0, 1)

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_BYTES = 2
WIRE_FIXED32 = 5

METADATA_MESH = 1
METADATA_BASIS = 2
METADATA_SOURCE = 3
MESH_VERTICES = 3
MESH_INDICES = 4
BASIS_LANDMARK = 1
BASIS_WEIGHT = 2


def _varint(data: bytes, pos: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, pos


def _fields(data: bytes) -> Iterator[Tuple[int, int, object]]:
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == WIRE_VARINT:
            value, pos = _varint(data, pos)
        elif wire == WIRE_BYTES:
            length, pos = _varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire == WIRE_FIXED32:
            value, pos = data[pos : pos + 4], pos + 4
        elif wire == WIRE_FIXED64:
            value, pos = data[pos : pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _repeated_floats(wire: int, value) -> List[float]:
    if wire == WIRE_BYTES:
        return list(struct.unpack(f"<{len(value) // 4}f", value))
    return [struct.unpack("<f", value)[0]]


def _repeated_ints(wire: int, value) -> List[int]:
    if wire != WIRE_BYTES:
        return [value]
    ints, pos = [], 0
    while pos < len(value):
        item, pos = _varint(value, pos)
        ints.append(item)
    return ints


def _procrustes(source: np.ndarray, target: np.ndarray, sqrt_weights: np.ndarray) -> np.ndarray:
    weighted_source = source * sqrt_weights
    weighted_target = target * sqrt_weights
    total_weight = float(np.sum(sqrt_weights * sqrt_weights))
    center = np.sum(weighted_source * sqrt_weights, axis=1) / total_weight
    centered = weighted_source - np.outer(center, sqrt_weights)

    u, _, vt = np.linalg.svd(weighted_target @ centered.T)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        u[:, 2] *= -1.0
    rotation = u @ vt
    scale = np.sum((rotation @ centered) * weighted_target) / np.sum(centered * weighted_source)
    rotation_and_scale = scale * rotation
    translation = (
        np.sum((weighted_target - rotation_and_scale @ weighted_source) * sqrt_weights, axis=1)
        / total_weight
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_and_scale
    transform[:3, 3] = translation
    return transform


class FaceGeometry:
    """Canonical face of the model bundle and the head pose fit against it."""

    def __init__(self, metadata: bytes):
        vertices: List[float] = []
        indices: List[int] = []
        weights: Dict[int, float] = {}
        source = 0
        for field, wire, value in _fields(metadata):
            if field == METADATA_SOURCE:
                source = value
            elif field == METADATA_MESH:
                for mesh_field, mesh_wire, mesh_value in _fields(value):
                    if mesh_field == MESH_VERTICES:
                        vertices.extend(_repeated_floats(mesh_wire, mesh_value))
                    elif mesh_field == MESH_INDICES:
                        indices.extend(_repeated_ints(mesh_wire, mesh_value))
            elif field == METADATA_BASIS:
                entry = {f: v for f, _, v in _fields(value)}
                weights[entry.get(BASIS_LANDMARK, 0)] = struct.unpack(
                    "<f", entry.get(BASIS_WEIGHT, bytes(4))
                )[0]
        if source not in FACE_LANDMARK_SOURCES:
            raise ValueError(f"geometry metadata expects input source {source}, not face landmarks")

        self._canonical = np.array(vertices, dtype=np.float64).reshape(-1, VERTEX_SIZE)[:, :3].T
        landmark_weights = np.zeros(self._canonical.shape[1])
        for landmark, weight in weights.items():
            landmark_weights[landmark] = weight
        self._sqrt_weights = np.sqrt(landmark_weights)
        triangles = np.array(indices, dtype=np.int64).reshape(-1, 3)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        self.mesh_edges = np.unique(edges, axis=0)

    def pose(self, landmarks: np.ndarray, image_size: ImageSize) -> Optional[np.ndarray]:
        """Transform from the canonical face to the camera frame in centimetres, or None for a degenerate face."""
        screen = landmarks[:MESH_LANDMARKS].T.astype(np.float64)
        offsets = screen[:2] - screen[:2].mean(axis=1, keepdims=True)
        if math.sqrt(float(np.max(np.sum(offsets * offsets, axis=0)))) <= MIN_FACE_EXTENT:
            return None

        height_at_near = 2.0 * NEAR_PLANE * math.tan(0.5 * math.radians(VERTICAL_FOV_DEG))
        width_at_near = image_size[0] * height_at_near / image_size[1]
        screen[1] = 1.0 - screen[1]
        screen *= np.array([[width_at_near], [height_at_near], [width_at_near]])
        screen[0] -= 0.5 * width_at_near
        screen[1] -= 0.5 * height_at_near
        depth_offset = float(screen[2].mean())

        mirrored = screen * np.array([[1.0], [1.0], [-1.0]])
        first_scale = self._scale(mirrored)
        second_scale = self._scale(self._unproject(screen, depth_offset, first_scale))
        metric = self._unproject(screen, depth_offset, first_scale * second_scale)
        return _procrustes(self._canonical, metric, self._sqrt_weights)

    def _scale(self, points: np.ndarray) -> float:
        return float(np.linalg.norm(_procrustes(self._canonical, points, self._sqrt_weights)[:3, 0]))

    @staticmethod
    def _unproject(screen: np.ndarray, depth_offset: float, scale: float) -> np.ndarray:
        depth = (screen[2] - depth_offset + NEAR_PLANE) / scale
        return np.stack([screen[0] * depth / NEAR_PLANE, screen[1] * depth / NEAR_PLANE, -depth])
