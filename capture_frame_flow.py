"""
Provide one shared frame flow for capture scripts.
"""

from typing import Any

import cv2
import mediapipe as mp


def detect_face_landmarker_result(landmarker: Any, frame_bgr: Any) -> Any:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect(mp_image)


def finalize_ui_frame(frame_bgr: Any, mirror_view: bool = True) -> Any:
    if mirror_view:
        return cv2.flip(frame_bgr, 1)
    return frame_bgr
