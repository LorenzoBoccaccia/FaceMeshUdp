"""
Camera capture in the cheapest mode that still gives the face landmark model full detail.

The face landmark model reads the face through a fixed 256x256 crop, so frames larger than
needed only add read and copy time, while frames shorter than 768 lines make it upsample the face
and blur the landmarks. Modes are tried from 1024x768 upward, preferring NV12, which the driver
scales in hardware, and 4:3 framing, which gives the face more of the frame. A mode is accepted
only when the camera delivers it at the requested size and at full frame rate; DirectShow is used
only at its native size, where it does not scale on the CPU.
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_CANDIDATE_MODES: List[Tuple[str, Optional[str], int, int, int, str]] = [
    ("msmf",  "NV12", 1024,  768, 30, "MSMF NV12 1024x768"),
    ("msmf",  "YUY2", 1024,  768, 30, "MSMF YUY2 1024x768"),
    ("msmf",  "NV12", 1280,  960, 30, "MSMF NV12 1280x960"),
    ("msmf",  "YUY2", 1280,  960, 30, "MSMF YUY2 1280x960"),
    ("msmf",  "NV12", 1920, 1080, 30, "MSMF NV12 1920x1080"),
    ("msmf",  "NV12", 2560, 1440, 30, "MSMF NV12 2560x1440"),
    ("msmf",  None,      0,    0,  0, "MSMF native"),
    ("dshow", None,      0,    0,  0, "DShow native"),
    ("any",   None,      0,    0,  0, "any backend native"),
]

_BACKEND_MAP = {
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any":   None,
}

_MAX_READ_MS = 60.0


def _fourcc(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip().strip("\x00").upper()


class CameraReader:
    """Reads frames from a camera device, picking the most CPU-efficient mode."""

    def __init__(self, camera_id: int = 0):
        self._camera_id = camera_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self.pixel_format: str = "bgr"
        self.fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0

    def _probe_format(self, cap: cv2.VideoCapture, probe_frame: np.ndarray) -> str:
        """Name the pixel layout of the frames the camera delivers in the negotiated mode."""
        if probe_frame is None:
            return "bgr"

        fourcc = _fourcc(cap)
        channels = probe_frame.shape[2] if probe_frame.ndim == 3 else 1

        if channels == 1:
            if fourcc == "NV12":
                return "nv12"
            if fourcc in ("YUY2", "YUYV"):
                return "yuyv"
            return "gray"

        return "bgr"

    def _try_candidate(
        self,
        backend_name: str,
        fourcc: Optional[str],
        width: int,
        height: int,
        fps: int,
        label: str,
    ) -> Optional[Tuple[cv2.VideoCapture, dict]]:
        """Open the camera in a candidate mode if it delivers that mode at full frame rate.

        Returns the (cap, info) pair when the camera opens, keeps the requested size and reads
        within one frame period of a 30 fps camera, and None otherwise.
        """
        backend_const = _BACKEND_MAP.get(backend_name, None)
        cap = (
            cv2.VideoCapture(self._camera_id, backend_const)
            if backend_const is not None
            else cv2.VideoCapture(self._camera_id)
        )
        if not cap.isOpened():
            cap.release()
            logger.info("CameraReader: %s — open failed", label)
            return None

        if fourcc and len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        ok, probe = cap.read()
        if not ok:
            cap.release()
            logger.info("CameraReader: %s — first read failed", label)
            return None

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > 0 and height > 0 and (actual_w != width or actual_h != height):
            cap.release()
            logger.info(
                "CameraReader: %s — dims rejected (got %dx%d, wanted %dx%d)",
                label, actual_w, actual_h, width, height,
            )
            return None

        timings: List[float] = []
        for _ in range(4):
            t0 = time.perf_counter()
            ok, _ = cap.read()
            if not ok:
                cap.release()
                logger.info("CameraReader: %s — sustained read failed", label)
                return None
            timings.append(time.perf_counter() - t0)
        timings.sort()
        median_ms = timings[len(timings) // 2] * 1000.0
        if median_ms > _MAX_READ_MS:
            cap.release()
            logger.info(
                "CameraReader: %s — median read %.0f ms exceeds %.0f ms",
                label, median_ms, _MAX_READ_MS,
            )
            return None

        pixel_format = self._probe_format(cap, probe)

        info = {
            "backend": backend_name,
            "candidate": label,
            "index": self._camera_id,
            "width": actual_w,
            "height": actual_h,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": _fourcc(cap),
            "pixel_format": pixel_format,
            "median_read_ms": median_ms,
        }
        return cap, info

    def _open_camera(self) -> Tuple[cv2.VideoCapture, dict]:
        """Walk the candidate ladder, accept the first that works."""
        for cand in _CANDIDATE_MODES:
            result = self._try_candidate(*cand)
            if result is not None:
                return result
        raise RuntimeError(
            "Unable to open camera with any of the candidate modes"
        )

    def open(self) -> dict:
        """Open the camera and return info dict."""
        self._cap, info = self._open_camera()
        self.pixel_format = info["pixel_format"]
        self.fps = float(info.get("fps", 0.0) or 0.0)
        self._frame_width = int(info.get("width", 0) or 0)
        self._frame_height = int(info.get("height", 0) or 0)
        logger.info(
            f"CameraReader: accepted '{info.get('candidate', '?')}' — "
            f"backend={info['backend']} index={info['index']} "
            f"{info['width']}x{info['height']} {info['fps']:.1f}fps "
            f"fourcc={info.get('fourcc', '????')!r} pixel_format={self.pixel_format} "
            f"median_read={info.get('median_read_ms', 0.0):.1f}ms",
        )
        logger.info("CameraReader: Webcam capture started.")
        return info

    def read_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """Read one frame from the camera and return it with a timestamp.

        Returns:
            Tuple of (frame, timestamp_ms). Frame is None on failure.
        """
        if self._cap is None:
            return None, 0

        ok, frame = self._cap.read()
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                logger.warning("CameraReader: cap.read() returned False")
            elif self._consecutive_failures % 100 == 0:
                logger.warning(
                    f"CameraReader: cap.read() has failed {self._consecutive_failures} consecutive times"
                )
            return None, 0

        if (
            self.pixel_format == "nv12"
            and self._frame_height > 0
            and self._frame_width > 0
            and frame is not None
        ):
            expected = self._frame_height * self._frame_width * 3 // 2
            if frame.size == expected:
                frame = frame.reshape(self._frame_height * 3 // 2, self._frame_width)

        if self._consecutive_failures > 0:
            logger.info(
                f"CameraReader: cap.read() recovered after {self._consecutive_failures} failures"
            )
            self._consecutive_failures = 0

        return frame, int(time.time() * 1000)

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("CameraReader: Released.")
