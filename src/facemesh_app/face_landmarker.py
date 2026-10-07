"""Track one face through a video and measure its landmarks, head pose and expression.

Runs the networks of the MediaPipe face landmarker bundle (face_landmarker.task, Apache-2.0) with
OpenCV, wired as MediaPipe's face landmarker graph does: the detector finds the face, then in a
video each frame's landmarks place the crop for the next one.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .face_blendshapes import FaceBlendshapes
from .face_detector import FaceDetector
from .face_geometry import FaceGeometry
from .face_region import FaceRegion, ImageSize
from .landmark_smoothing import LandmarkSmoother

logger = logging.getLogger(__name__)

BUNDLE_PATH = Path("face_landmarker.task")
BUNDLE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
DETECTOR_MODEL = "face_detector.tflite"
LANDMARKS_MODEL = "face_landmarks_detector.tflite"
BLENDSHAPES_MODEL = "face_blendshapes.tflite"
GEOMETRY_METADATA = "geometry_pipeline_metadata_landmarks.binarypb"

INPUT_SIZE = 256
VALUE_RANGE = (0.0, 1.0)
NUM_LANDMARKS = 478
LANDMARK_OUTPUTS = ("Identity", "Identity_1")
MIN_PRESENCE = 0.5
TRACKING_SCALE = 1.5
ROTATION_START_LANDMARK = 33
ROTATION_END_LANDMARK = 263


def ensure_bundle() -> Path:
    """Path of the face landmarker bundle, downloaded on first use."""
    if not BUNDLE_PATH.exists():
        logger.info(f"Downloading the face landmarker model from {BUNDLE_URL}")
        urllib.request.urlretrieve(BUNDLE_URL, str(BUNDLE_PATH))
    return BUNDLE_PATH


@dataclass(frozen=True)
class FaceLandmarks:
    """The face in one frame.

    Landmarks are x and y normalized to the image width and height, z on the scale of x. The
    transform maps the canonical face to the camera frame in centimetres. Mesh edges join the
    landmarks into the face surface.
    """

    landmarks: np.ndarray
    transform: Optional[np.ndarray]
    blendshapes: Optional[Dict[str, float]] = None
    mesh_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))

    def __post_init__(self):
        for values in (self.landmarks, self.transform, self.mesh_edges):
            if values is not None:
                values.setflags(write=False)


def _load_net(model: bytes) -> cv2.dnn.Net:
    return cv2.dnn.readNetFromTFLite(np.frombuffer(model, dtype=np.uint8))


class FaceLandmarker:
    """The face of a video stream frame after frame, or of a single image."""

    def __init__(
        self,
        detector: FaceDetector,
        landmarks_net: cv2.dnn.Net,
        geometry: FaceGeometry,
        blendshapes: Optional[FaceBlendshapes],
    ):
        self._detector = detector
        self._landmarks_net = landmarks_net
        self._geometry = geometry
        self._blendshapes = blendshapes
        self._smoother = LandmarkSmoother()
        self._tracked: Optional[FaceRegion] = None

    @classmethod
    def from_bundle(cls, path: Path, with_blendshapes: bool) -> FaceLandmarker:
        """Landmarker running the models of a face landmarker bundle."""
        with zipfile.ZipFile(path) as bundle:
            detector = FaceDetector(_load_net(bundle.read(DETECTOR_MODEL)))
            landmarks_net = _load_net(bundle.read(LANDMARKS_MODEL))
            geometry = FaceGeometry(bundle.read(GEOMETRY_METADATA))
            blendshapes = (
                FaceBlendshapes(_load_net(bundle.read(BLENDSHAPES_MODEL))) if with_blendshapes else None
            )
        return cls(detector, landmarks_net, geometry, blendshapes)

    def track(self, image: np.ndarray, timestamp_ms: int) -> Optional[FaceLandmarks]:
        """The face in this RGB video frame, or None when there is none; frames must come in time order."""
        image_size: ImageSize = (image.shape[1], image.shape[0])
        region = self._tracked or self._detector.find(image)
        landmarks = self._landmarks_in(image, region) if region is not None else None
        if landmarks is None:
            self._tracked = None
            self._smoother.reset()
            return None
        self._tracked = FaceRegion.from_box(
            landmarks[:, :2].min(axis=0),
            landmarks[:, :2].max(axis=0),
            landmarks[ROTATION_START_LANDMARK],
            landmarks[ROTATION_END_LANDMARK],
            image_size,
        ).scaled(TRACKING_SCALE, image_size, square=True)
        return self._measure(self._smoother.smooth(landmarks, timestamp_ms, image_size), image_size)

    def measure(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """The face in a single RGB image, found afresh, neither tracked nor smoothed."""
        region = self._detector.find(image)
        landmarks = self._landmarks_in(image, region) if region is not None else None
        if landmarks is None:
            return None
        return self._measure(landmarks, (image.shape[1], image.shape[0]))

    def _measure(self, landmarks: np.ndarray, image_size: ImageSize) -> FaceLandmarks:
        return FaceLandmarks(
            landmarks=landmarks,
            transform=self._geometry.pose(landmarks, image_size),
            blendshapes=self._blendshapes.scores(landmarks, image_size) if self._blendshapes else None,
            mesh_edges=self._geometry.mesh_edges,
        )

    def _landmarks_in(self, image: np.ndarray, region: FaceRegion) -> Optional[np.ndarray]:
        crop = region.crop(image, INPUT_SIZE, VALUE_RANGE, cv2.BORDER_REPLICATE)
        self._landmarks_net.setInput(crop.blob)
        raw, presence_logit = self._landmarks_net.forward(list(LANDMARK_OUTPUTS))
        presence = 1.0 / (1.0 + np.exp(-float(presence_logit.reshape(-1)[0])))
        if not presence > MIN_PRESENCE:
            return None
        points = raw.reshape(NUM_LANDMARKS, 3).astype(np.float64) / INPUT_SIZE
        return np.column_stack([crop.to_image(points[:, :2]), points[:, 2] * crop.z_scale])
