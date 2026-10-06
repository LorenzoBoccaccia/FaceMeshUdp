"""
Pipeline steps for face processing.
Each step processes data and passes it to the next step in the pipeline.
"""

import logging
import math
import socket
import struct
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from .calibration import PERSON_AXES, CalibratedFaceAndGazeEvent, GazeModel
from .facemesh_dao import MM_PER_CM, FaceMeshEvent

logger = logging.getLogger(__name__)

BLINK_RECOVERY_RATIO = 0.95
EYE_OPENING_SETTLED = 0.005
OPEN_WIDTH_FRAMES = 10


class FaceMeshStep:
    """First pipeline step: Extract face mesh data from frames using MediaPipe FaceLandmarker."""

    _CONVERT_MAP = {
        "bgr": cv2.COLOR_BGR2RGB,
        "yuyv": cv2.COLOR_YUV2RGB_YUY2,
        "nv12": cv2.COLOR_YUV2RGB_NV12,
    }

    def __init__(self, face_landmarker: vision.FaceLandmarker):
        self.face_landmarker = face_landmarker
        self._last_timestamp_ms = -1

    def receive_frame(
        self, frame, timestamp_ms: int, pixel_format: str = "bgr"
    ) -> Optional[FaceMeshEvent]:
        if frame is None:
            logger.warning("Received None frame in FaceMeshStep")
            return None

        try:
            if pixel_format == "rgb":
                frame_rgb = frame
            else:
                code = self._CONVERT_MAP.get(pixel_format, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.cvtColor(frame, code)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(timestamp_ms)
            if ts <= self._last_timestamp_ms:
                ts = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts

            result = self.face_landmarker.detect_for_video(mp_image, ts)
            evt = FaceMeshEvent.from_landmarker_result(
                result,
                image_size=(frame_rgb.shape[1], frame_rgb.shape[0]),
                ts=ts,
            )
            return evt

        except Exception as e:
            logger.error(f"Error processing frame in FaceMeshStep: {e}")
            return None


class CalibrationAdapterStep:
    """Second pipeline step: Turn a face measurement into the gaze the calibration model derives from it."""

    def __init__(self, model: Optional[GazeModel] = None):
        self._model = model

    def set_model(self, model: GazeModel) -> None:
        self._model = model

    def receive_frame(
        self, frame: np.ndarray, face_mesh_event: Optional[FaceMeshEvent]
    ) -> Optional[CalibratedFaceAndGazeEvent]:
        """Calibrated gaze for the frame, or None without a calibration or a usable face."""
        if self._model is None:
            return None
        gaze = self._model.gaze(face_mesh_event)
        if gaze is None:
            return None
        return CalibratedFaceAndGazeEvent(face_mesh_event, self._model, gaze)


class BlinkRejectionStep:
    """Pipeline step: Withhold gaze while the eyelids close and reopen, when the iris reading is not gaze.

    A blink starts when the eye opening falls below the calibrated blink level and lasts until
    the lid is back near its open width before the blink, or has stopped reopening, which
    happens lower when the user looks down. The open width is the widest of the last open
    frames, so the closing lid cannot lower it.
    """

    def __init__(self):
        self._blinking = False
        self._open_openings: Deque[float] = deque(maxlen=OPEN_WIDTH_FRAMES)
        self._previous_opening: Optional[float] = None

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> Optional[CalibratedFaceAndGazeEvent]:
        """The calibrated event, or None while the user is blinking."""
        if calibrated_event is None:
            return None
        opening = calibrated_event.face_mesh_event.eye_opening
        blink_level = calibrated_event.model.blink_opening
        previous = self._previous_opening
        self._previous_opening = opening
        if self._blinking:
            reopened = bool(self._open_openings) and (
                opening >= BLINK_RECOVERY_RATIO * max(self._open_openings)
            )
            settled = (
                opening >= blink_level
                and previous is not None
                and opening - previous <= EYE_OPENING_SETTLED
            )
            self._blinking = not (reopened or settled)
        elif opening < blink_level:
            self._blinking = True
        else:
            self._open_openings.append(opening)
        return None if self._blinking else calibrated_event


@dataclass(frozen=True)
class GazeDirection:
    """Gaze direction handed to output steps, in degrees, opentrack axis convention."""

    yaw: float
    pitch: float


class GazeSmoothingStep:
    """Pipeline step: Steady the calibrated gaze with the mean of a trailing time window.

    Movement within the threshold, measured against the calibrated eye noise, is treated as
    fixation jitter and averaged away. When the gaze leaves the current fixation for good, the
    previous fixation's samples older than a short tail are blanked in place: the window keeps
    its length and refills with new frames, so the view eases into the new fixation instead of
    snapping. The output depends only on the samples inside the window, so a held gaze always
    settles on its calibrated absolute direction no matter how fast or slow it got there.
    """

    def __init__(
        self,
        window_ms: int = 0,
        reset_window_ms: Optional[int] = None,
        reset_threshold_sigma: float = 3.0,
    ):
        """Initialize gaze smoothing step.

        Args:
            window_ms: Length of the averaging window in milliseconds; 0 forwards the gaze raw
            reset_window_ms: Tail of the previous fixation kept after a jump; None disables jumps
            reset_threshold_sigma: Distance from the current fixation, in multiples of the
                calibrated eye noise, that counts as leaving it
        """
        self.window_ms = window_ms
        self.reset_window_ms = reset_window_ms
        self.reset_threshold_sigma = reset_threshold_sigma
        self.jump_count = 0
        self._samples: Deque[List[float]] = deque()
        self._fixation_ts = 0
        self._candidate: Optional[List[float]] = None

        logger.debug(
            f"GazeSmoothingStep initialized: window_ms={window_ms}, "
            f"reset_window_ms={reset_window_ms}, reset_threshold_sigma={reset_threshold_sigma}"
        )

    def _trim(self, ts: int) -> None:
        while self._samples and self._samples[0][0] <= ts - self.window_ms:
            self._samples.popleft()

    @staticmethod
    def _gazes(samples) -> np.ndarray:
        values = np.array([sample[1:] for sample in samples], dtype=float).reshape(-1, 2)
        return values[~np.isnan(values[:, 0])]

    @staticmethod
    def _distance(a, b, noise: Tuple[float, float]) -> float:
        return math.hypot((a[0] - b[0]) / noise[0], (a[1] - b[1]) / noise[1])

    def _drop_candidate(self) -> None:
        if (
            self._candidate is not None
            and self._samples
            and self._samples[-1] is self._candidate
        ):
            self._samples.pop()
        self._candidate = None

    def _start_fixation(self, ts: int) -> None:
        left_since = self._fixation_ts
        self._fixation_ts = ts
        left = [
            s[0] for s in self._samples if left_since <= s[0] < ts and not math.isnan(s[1])
        ]
        if not left:
            return
        last = left[-1]
        for sample in self._samples:
            if sample[0] >= left_since and last - sample[0] >= self.reset_window_ms:
                sample[1] = sample[2] = math.nan

    def _accept(self, sample: List[float], noise: Tuple[float, float]) -> None:
        fixation = self._gazes(
            entry
            for entry in self._samples
            if entry[0] >= self._fixation_ts and entry is not self._candidate
        )
        if not len(fixation):
            self._drop_candidate()
            self._start_fixation(sample[0])
        else:
            gaze = sample[1:]
            off_fixation = self._distance(gaze, np.median(fixation, axis=0), noise)
            candidate = self._candidate
            if off_fixation <= self.reset_threshold_sigma:
                self._drop_candidate()
            elif candidate is not None and self._distance(gaze, candidate[1:], noise) < off_fixation:
                self._candidate = None
                self._start_fixation(candidate[0])
                self.jump_count += 1
            else:
                self._drop_candidate()
                self._candidate = sample
        self._samples.append(sample)

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> Optional[GazeDirection]:
        """Produce the gaze direction to forward for this frame.

        Args:
            frame: Input frame (not used but kept for interface consistency)
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)

        Returns:
            Mean gaze over the window's valid samples, or None when the frame has no calibrated gaze
        """
        if calibrated_event is None:
            self._drop_candidate()
            return None

        yaw = calibrated_event.gaze.yaw
        pitch = calibrated_event.gaze.pitch

        if self.window_ms <= 0:
            return GazeDirection(yaw=yaw, pitch=pitch)

        ts = calibrated_event.face_mesh_event.ts
        sample = [ts, yaw, pitch]
        self._trim(ts)
        if self.reset_window_ms is None:
            self._samples.append(sample)
        else:
            self._accept(sample, calibrated_event.model.eye_noise_deg)

        mean_yaw, mean_pitch = self._gazes(self._samples).mean(axis=0)
        return GazeDirection(yaw=float(mean_yaw), pitch=float(mean_pitch))


class OpenTrackForwardStep:
    """Final pipeline step: Forward calibrated face and gaze data to opentrack's UDP tracker input.

    It is disabled by default and can be enabled when opentrack is the consumer of the pose stream.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 4242, enabled: bool = False
    ):
        """Initialize OpenTrack forward step.

        Args:
            host: opentrack host address (default: "127.0.0.1")
            port: opentrack UDP tracker port (default: 4242)
            enabled: Whether OpenTrack forwarding is active (default: False)
        """
        self.host = host
        self.port = port
        self.enabled = enabled
        self._socket = None

        # Initialize socket if enabled
        if self.enabled:
            self._create_socket()

        logger.debug(
            f"OpenTrackForwardStep initialized: host={host}, port={port}, enabled={enabled}"
        )

    def _create_socket(self) -> None:
        """Create UDP socket for sending data."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)  # Non-blocking mode
            logger.debug(f"UDP socket created for {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Failed to create UDP socket: {e}")
            self._socket = None

    def _close_socket(self) -> None:
        """Close UDP socket."""
        if self._socket is not None:
            try:
                self._socket.close()
                logger.debug("UDP socket closed")
            except Exception as e:
                logger.warning(f"Error closing UDP socket: {e}")
            finally:
                self._socket = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable OpenTrack forwarding.

        Args:
            enabled: Whether to enable OpenTrack forwarding
        """
        if self.enabled == enabled:
            return

        self.enabled = enabled

        if enabled:
            self._create_socket()
        else:
            self._close_socket()

        logger.debug(f"OpenTrackForwardStep enabled: {enabled}")

    def _serialize_event(
        self, event: CalibratedFaceAndGazeEvent, gaze: GazeDirection
    ) -> bytes:
        """Serialize calibrated event to OpenTrack UDP payload.

        Args:
            event: Calibrated face and gaze event
            gaze: Gaze direction to forward as yaw and pitch

        Returns:
            Binary OpenTrack pose payload: eye position in centimetres to the user's right,
            up and away from the camera, then yaw, pitch and roll in degrees
        """
        face_event = event.face_mesh_event
        right, up, distance = PERSON_AXES.T @ face_event.eye_position / MM_PER_CM
        return struct.pack(
            "<6d", right, up, distance, gaze.yaw, gaze.pitch, face_event.roll or 0.0
        )

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
        gaze: Optional[GazeDirection],
    ) -> None:
        """Forward calibrated data to opentrack.

        Args:
            frame: Input frame (not used but kept for interface consistency)
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)
            gaze: Gaze direction to forward (optional)
        """
        if not self.enabled:
            return

        if calibrated_event is None or gaze is None:
            logger.debug("No calibrated gaze, skipping OpenTrack forward")
            return

        if self._socket is None:
            logger.warning("UDP socket is None, skipping OpenTrack forward")
            return

        try:
            message_bytes = self._serialize_event(calibrated_event, gaze)

            self._socket.sendto(message_bytes, (self.host, self.port))

            logger.debug(
                f"OpenTrack pose sent to {self.host}:{self.port}: {len(message_bytes)} bytes"
            )

        except (socket.error, OSError) as e:
            logger.warning(f"Socket error sending OpenTrack pose: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error sending OpenTrack pose: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending OpenTrack pose: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._close_socket()
