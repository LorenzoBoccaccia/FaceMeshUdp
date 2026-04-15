"""
Compatibility exports for overlay modules.
"""

from typing import Dict, Optional

import numpy as np

from .overlay_calibration import CalibrationOverlayManager
from .overlay_common import get_display_geo
from .overlay_runtime import RuntimeOverlayManager


class OverlayManager:
    """Provide backward-compatible overlay manager selection."""

    def __init__(
        self,
        display: Dict,
        capture_enabled: bool = False,
        overlay_fps: int = 60,
        calibration_mode: bool = False,
        click_through: bool = False,
    ):
        if calibration_mode:
            self._impl = CalibrationOverlayManager(
                display,
                overlay_fps=overlay_fps,
            )
        else:
            self._impl = RuntimeOverlayManager(
                display,
                capture_enabled=capture_enabled,
                overlay_fps=overlay_fps,
                click_through=click_through,
            )

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def render_mesh(
        self, evt: Optional[Dict], capture_live_frame: Optional[np.ndarray] = None
    ):
        if isinstance(self._impl, RuntimeOverlayManager):
            return self._impl.render_mesh(evt, capture_live_frame)
        return self._impl.render_mesh(evt)
