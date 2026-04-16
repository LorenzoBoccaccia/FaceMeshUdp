"""
Calibration overlay window for 9-point calibration flow.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import pygame

from .calibration import CalibrationPoint
from .facemesh_dao import safe_float
from .gaze_primitives import screen_xy_to_head_angles
from .overlay_common import (
    BLACK,
    BLUE,
    DOT_RADIUS,
    GREEN,
    RED,
    WHITE,
    set_window_topmost,
)


CALIB_INSET = 50
CALIB_BLINK_MIN_MS = 600
CALIB_COUNTDOWN_MS = 1000
CALIB_AVG_MS = 1000
CALIB_BLINK_PERIOD_MS = 220
CALIB_ALIGNMENT_WINDOW_MS = 500
CALIB_ALIGNMENT_HOLD_MS = 400
CALIB_ALIGNMENT_STABILITY_DEG = 1.0
CALIB_ALIGNMENT_TOLERANCE_DEG = 3.0
CALIB_ALIGNMENT_RING_SCALE = 1.5
CALIB_ARROW_PIXELS_PER_DEG = 30.0
CALIB_ARROW_MIN_DEG = 0.3
CALIB_ARROW_MAX_PX = 260.0


def _signum(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


class CalibrationOverlayManager:
    """Manage calibration overlay window and calibration state machine."""

    def __init__(
        self,
        display: Dict,
        overlay_fps: int = 60,
    ):
        self._display = display
        self._overlay_fps = overlay_fps
        self._width = int(display["width"])
        self._height = int(display["height"])
        self._width_mm = float(display.get("width_mm", 0) or 0.0)
        self._height_mm = float(display.get("height_mm", 0) or 0.0)
        if self._width_mm <= 0.0 or self._height_mm <= 0.0:
            fallback_mm_per_px = 0.25
            self._width_mm = self._width * fallback_mm_per_px
            self._height_mm = self._height * fallback_mm_per_px

        self._screen = None
        self._clock = None
        self._font = None
        self._hwnd = None

        self._running = False
        self._should_exit = False

        self._calibration_sequence: List[Dict] = []
        self._current_calib_idx: int = 0
        self._calib_phase: str = "idle"
        self._calib_phase_start: int = 0
        self._calib_samples: List[Dict] = []
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._recent_head_samples: List[Tuple[int, float, float, float, float, float]] = []
        self._alignment_start_ms: Optional[int] = None
        self._alignment_active: bool = False
        self._c_baseline_yaw: Optional[float] = None
        self._c_baseline_pitch: Optional[float] = None
        self._c_head_x: Optional[float] = None
        self._c_head_y: Optional[float] = None
        self._c_head_z: Optional[float] = None
        self._screen_center_cam_x: Optional[float] = None
        self._screen_center_cam_y: Optional[float] = None
        self._alignment_error_yaw: Optional[float] = None
        self._alignment_error_pitch: Optional[float] = None

    def initialize(self):
        """Create and configure calibration overlay window."""
        pygame.init()
        pygame.font.init()
        os.environ.setdefault(
            "SDL_VIDEO_WINDOW_POS", f"{self._display['x']},{self._display['y']}"
        )
        self._screen = pygame.display.set_mode(
            (self._width, self._height), pygame.NOFRAME
        )
        pygame.display.set_caption("FaceMesh Calibration")
        self._hwnd = pygame.display.get_wm_info().get("window")
        set_window_topmost(self._hwnd)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 34)
        self._running = True

    def shutdown(self):
        """Close calibration overlay resources."""
        if self._screen:
            pygame.quit()
        self._running = False

    def handle_events(self):
        """Process calibration window events."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._should_exit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._should_exit = True

    def render_mesh(self, evt: Optional[Dict]):
        """Render one calibration frame."""
        _ = evt
        self._screen.fill(BLACK)
        if self._calib_phase != "idle":
            current_point = self.get_current_calib_point()
            if current_point:
                current_time = int(time.time() * 1000)
                elapsed_ms = current_time - self._calib_phase_start
                self.render_calibration(current_point, self._calib_phase, elapsed_ms)
        pygame.display.update()
        self._clock.tick(max(1, int(self._overlay_fps)))

    def clear(self):
        """Clear calibration overlay surface."""
        self._screen.fill(BLACK)

    def update(self):
        """Flip buffers and enforce target frame rate."""
        pygame.display.update()
        if self._clock:
            self._clock.tick(max(1, int(self._overlay_fps)))

    def start_calibration_sequence(self, width: float, height: float):
        """Initialize calibration sequence targets and reset state."""
        self._calibration_sequence = self._make_calib_seq(width, height)
        self._current_calib_idx = 0
        self._calib_phase = "blink_pre"
        self._calib_phase_start = int(time.time() * 1000)
        self._calib_samples = []
        self._center_x = float(width) / 2.0
        self._center_y = float(height) / 2.0
        self._recent_head_samples = []
        self._alignment_start_ms = None
        self._alignment_active = False
        self._c_baseline_yaw = None
        self._c_baseline_pitch = None
        self._c_head_x = None
        self._c_head_y = None
        self._c_head_z = None
        self._screen_center_cam_x = None
        self._screen_center_cam_y = None
        self._alignment_error_yaw = None
        self._alignment_error_pitch = None

    def get_current_calib_point(self) -> Optional[Dict]:
        """Return current calibration target."""
        if 0 <= self._current_calib_idx < len(self._calibration_sequence):
            return self._calibration_sequence[self._current_calib_idx]
        return None

    def get_calibration_phase(self) -> str:
        """Return current calibration phase."""
        return self._calib_phase

    def _evaluate_alignment(
        self, evt: Optional[Dict], current_time: int, current_point: Dict
    ) -> bool:
        """Return True when head pose is stable and pointing at the current target.

        For the center point (C) we capture whatever head pose the user
        presents — it defines both the camera→monitor angular offset and the
        screen-center position in camera-mm. For every other point we trust
        head_z (camera mm) and the corrected face angle, project the ray onto
        the screen plane at z=0, and compare to the target pixel converted to
        mm via the monitor's physical size reported by the OS.
        """
        if not evt:
            self._recent_head_samples.clear()
            self._alignment_error_yaw = None
            self._alignment_error_pitch = None
            return False
        head_yaw = safe_float(evt.get("head_yaw"), float("nan"))
        head_pitch = safe_float(evt.get("head_pitch"), float("nan"))
        head_x = safe_float(evt.get("head_x"), float("nan"))
        head_y = safe_float(evt.get("head_y"), float("nan"))
        head_z = safe_float(evt.get("head_z"), float("nan"))
        if not all(
            math.isfinite(v) for v in (head_yaw, head_pitch, head_x, head_y, head_z)
        ):
            self._recent_head_samples.clear()
            self._alignment_error_yaw = None
            self._alignment_error_pitch = None
            return False

        self._recent_head_samples.append(
            (current_time, head_yaw, head_pitch, head_x, head_y, head_z)
        )
        cutoff = current_time - CALIB_ALIGNMENT_WINDOW_MS
        self._recent_head_samples = [
            entry for entry in self._recent_head_samples if entry[0] >= cutoff
        ]
        if len(self._recent_head_samples) < 5:
            self._alignment_error_yaw = None
            self._alignment_error_pitch = None
            return False

        n = len(self._recent_head_samples)
        mean_yaw = sum(s[1] for s in self._recent_head_samples) / n
        mean_pitch = sum(s[2] for s in self._recent_head_samples) / n
        var_yaw = sum((s[1] - mean_yaw) ** 2 for s in self._recent_head_samples) / n
        var_pitch = (
            sum((s[2] - mean_pitch) ** 2 for s in self._recent_head_samples) / n
        )

        is_center = current_point.get("name") == "C"

        if is_center:
            self._alignment_error_yaw = None
            self._alignment_error_pitch = None
            error_yaw = 0.0
            error_pitch = 0.0
        elif (
            self._screen_center_cam_x is None
            or self._screen_center_cam_y is None
            or head_z <= 0.0
        ):
            self._alignment_error_yaw = None
            self._alignment_error_pitch = None
            return False
        else:
            nose_x = safe_float(current_point.get("nose_x"), self._center_x)
            nose_y = safe_float(current_point.get("nose_y"), self._center_y)
            angles = screen_xy_to_head_angles(
                screen_x=nose_x,
                screen_y=nose_y,
                head_x=head_x,
                head_y=head_y,
                head_z=head_z,
                center_zeta=head_z,
                screen_center_cam_x=self._screen_center_cam_x,
                screen_center_cam_y=self._screen_center_cam_y,
                screen_center_cam_z=0.0,
                screen_axis_x_x=1.0,
                screen_axis_x_y=0.0,
                screen_axis_x_z=0.0,
                screen_axis_y_x=0.0,
                screen_axis_y_y=1.0,
                screen_axis_y_z=0.0,
                screen_fit_rmse=0.0,
                screen_scale_x=float(self._width) / self._width_mm,
                screen_scale_y=float(self._height) / self._height_mm,
                origin_x=self._center_x,
                origin_y=self._center_y,
            )
            if angles is None:
                self._alignment_error_yaw = None
                self._alignment_error_pitch = None
                return False
            expected_yaw, expected_pitch = angles
            error_yaw = expected_yaw - head_yaw
            error_pitch = expected_pitch - head_pitch
            self._alignment_error_yaw = error_yaw
            self._alignment_error_pitch = error_pitch

        if math.sqrt(var_yaw) > CALIB_ALIGNMENT_STABILITY_DEG:
            return False
        if math.sqrt(var_pitch) > CALIB_ALIGNMENT_STABILITY_DEG:
            return False
        if is_center:
            return True
        if abs(error_yaw) > CALIB_ALIGNMENT_TOLERANCE_DEG:
            return False
        if abs(error_pitch) > CALIB_ALIGNMENT_TOLERANCE_DEG:
            return False
        return True

    def is_aligned(self) -> bool:
        """Expose alignment status so rendering can show confirmation ring."""
        return self._alignment_active

    def update_calibration_state(
        self, evt: Dict
    ) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Advance calibration state and emit points when sampling completes."""
        current_time = int(time.time() * 1000)
        elapsed_ms = current_time - self._calib_phase_start
        current_point = self.get_current_calib_point()
        fallback_zeta = max(self._width, self._height) * 0.75

        if current_point is None:
            return True, None

        if self._calib_phase == "blink_pre":
            aligned = self._evaluate_alignment(evt, current_time, current_point)
            if aligned:
                if self._alignment_start_ms is None:
                    self._alignment_start_ms = current_time
                self._alignment_active = True
                hold_ms = current_time - self._alignment_start_ms
                ready = (
                    hold_ms >= CALIB_ALIGNMENT_HOLD_MS
                    and elapsed_ms >= CALIB_BLINK_MIN_MS
                )
            else:
                self._alignment_start_ms = None
                self._alignment_active = False
                ready = False
            if ready:
                self._calib_phase = "countdown"
                self._calib_phase_start = current_time
                self._alignment_active = False
                self._alignment_start_ms = None
                self._recent_head_samples.clear()
                self._alignment_error_yaw = None
                self._alignment_error_pitch = None

        elif self._calib_phase == "countdown":
            if elapsed_ms >= CALIB_COUNTDOWN_MS:
                self._calib_phase = "sampling"
                self._calib_phase_start = current_time
                self._calib_samples = []

        elif self._calib_phase == "sampling":
            if evt:
                combined_yaw = evt.get("raw_combined_eye_gaze_yaw")
                combined_pitch = evt.get("raw_combined_eye_gaze_pitch")
                if combined_yaw is not None and combined_pitch is not None:
                    left_eye_yaw = evt.get("raw_left_eye_gaze_yaw")
                    left_eye_pitch = evt.get("raw_left_eye_gaze_pitch")
                    right_eye_yaw = evt.get("raw_right_eye_gaze_yaw")
                    right_eye_pitch = evt.get("raw_right_eye_gaze_pitch")
                    head_yaw = evt.get("head_yaw")
                    head_pitch = evt.get("head_pitch")
                    zeta = evt.get("zeta")
                    head_x = evt.get("head_x")
                    head_y = evt.get("head_y")
                    head_z = evt.get("head_z")
                    self._calib_samples.append(
                        {
                            "eye_yaw": combined_yaw,
                            "eye_pitch": combined_pitch,
                            "left_eye_yaw": (
                                left_eye_yaw if left_eye_yaw is not None else combined_yaw
                            ),
                            "left_eye_pitch": (
                                left_eye_pitch
                                if left_eye_pitch is not None
                                else combined_pitch
                            ),
                            "right_eye_yaw": (
                                right_eye_yaw
                                if right_eye_yaw is not None
                                else combined_yaw
                            ),
                            "right_eye_pitch": (
                                right_eye_pitch
                                if right_eye_pitch is not None
                                else combined_pitch
                            ),
                            "head_yaw": head_yaw,
                            "head_pitch": head_pitch,
                            "zeta": zeta,
                            "head_x": head_x,
                            "head_y": head_y,
                            "head_z": head_z,
                        }
                    )

            if elapsed_ms >= CALIB_AVG_MS:
                if self._calib_samples:
                    avg_yaw = sum(s["eye_yaw"] for s in self._calib_samples) / len(
                        self._calib_samples
                    )
                    avg_pitch = sum(s["eye_pitch"] for s in self._calib_samples) / len(
                        self._calib_samples
                    )
                    avg_left_yaw = sum(
                        s["left_eye_yaw"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_left_pitch = sum(
                        s["left_eye_pitch"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_right_yaw = sum(
                        s["right_eye_yaw"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_right_pitch = sum(
                        s["right_eye_pitch"] for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_yaw = sum(
                        safe_float(s.get("head_yaw"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_pitch = sum(
                        safe_float(s.get("head_pitch"), 0.0)
                        for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_zeta = sum(
                        safe_float(s.get("zeta"), fallback_zeta)
                        for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_x = sum(
                        safe_float(s.get("head_x"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_y = sum(
                        safe_float(s.get("head_y"), 0.0) for s in self._calib_samples
                    ) / len(self._calib_samples)
                    avg_head_z = sum(
                        safe_float(s.get("head_z"), avg_zeta)
                        for s in self._calib_samples
                    ) / len(self._calib_samples)

                    calib_point = CalibrationPoint(
                        name=current_point["name"],
                        screen_x=current_point["x"],
                        screen_y=current_point["y"],
                        raw_eye_yaw=avg_yaw,
                        raw_eye_pitch=avg_pitch,
                        raw_left_eye_yaw=avg_left_yaw,
                        raw_left_eye_pitch=avg_left_pitch,
                        raw_right_eye_yaw=avg_right_yaw,
                        raw_right_eye_pitch=avg_right_pitch,
                        sample_count=len(self._calib_samples),
                        head_yaw=avg_head_yaw,
                        head_pitch=avg_head_pitch,
                        zeta=avg_zeta,
                        head_x=avg_head_x,
                        head_y=avg_head_y,
                        head_z=avg_head_z,
                        nose_target_x=current_point.get("nose_x"),
                        nose_target_y=current_point.get("nose_y"),
                        eye_target_x=current_point.get("eye_x"),
                        eye_target_y=current_point.get("eye_y"),
                    )
                else:
                    head_yaw = safe_float(evt.get("head_yaw") if evt else None, 0.0)
                    head_pitch = safe_float(evt.get("head_pitch") if evt else None, 0.0)
                    zeta = safe_float(evt.get("zeta") if evt else None, fallback_zeta)
                    head_x = safe_float(evt.get("head_x") if evt else None, 0.0)
                    head_y = safe_float(evt.get("head_y") if evt else None, 0.0)
                    head_z = safe_float(evt.get("head_z") if evt else None, zeta)
                    calib_point = CalibrationPoint(
                        name=current_point["name"],
                        screen_x=current_point["x"],
                        screen_y=current_point["y"],
                        raw_eye_yaw=0.0,
                        raw_eye_pitch=0.0,
                        raw_left_eye_yaw=0.0,
                        raw_left_eye_pitch=0.0,
                        raw_right_eye_yaw=0.0,
                        raw_right_eye_pitch=0.0,
                        sample_count=0,
                        head_yaw=head_yaw,
                        head_pitch=head_pitch,
                        zeta=zeta,
                        head_x=head_x,
                        head_y=head_y,
                        head_z=head_z,
                        nose_target_x=current_point.get("nose_x"),
                        nose_target_y=current_point.get("nose_y"),
                        eye_target_x=current_point.get("eye_x"),
                        eye_target_y=current_point.get("eye_y"),
                    )

                if current_point["name"] == "C":
                    self._c_baseline_yaw = safe_float(calib_point.head_yaw, 0.0)
                    self._c_baseline_pitch = safe_float(calib_point.head_pitch, 0.0)
                    self._c_head_x = safe_float(calib_point.head_x, 0.0)
                    self._c_head_y = safe_float(calib_point.head_y, 0.0)
                    self._c_head_z = safe_float(
                        calib_point.head_z, safe_float(calib_point.zeta, 0.0)
                    )
                    self._screen_center_cam_x = self._c_head_x + self._c_head_z * math.tan(
                        math.radians(self._c_baseline_yaw)
                    )
                    self._screen_center_cam_y = self._c_head_y - self._c_head_z * math.tan(
                        math.radians(self._c_baseline_pitch)
                    )

                self._current_calib_idx += 1
                next_point = self.get_current_calib_point()
                if next_point is None:
                    return True, calib_point

                self._calib_phase = "blink_pre"
                self._calib_phase_start = current_time
                self._calib_samples = []
                self._recent_head_samples.clear()
                self._alignment_start_ms = None
                self._alignment_active = False
                self._alignment_error_yaw = None
                self._alignment_error_pitch = None
                return False, calib_point

        return False, None

    def render_calibration(
        self,
        current_point: Dict,
        phase: str,
        elapsed_ms: int,
    ):
        """Render calibration target for current phase."""
        aligned = phase == "blink_pre" and self._alignment_active
        if phase == "countdown":
            dot_color = RED
        elif phase == "sampling":
            dot_color = BLUE
        elif aligned:
            dot_color = WHITE
        else:
            if (elapsed_ms // CALIB_BLINK_PERIOD_MS) % 2 == 0:
                dot_color = WHITE
            else:
                dot_color = None

        if dot_color:
            nose_x = int(round(current_point["nose_x"]))
            nose_y = int(round(current_point["nose_y"]))
            eye_x = int(round(current_point["eye_x"]))
            eye_y = int(round(current_point["eye_y"]))
            nose_color = RED if phase != "sampling" else BLUE
            eye_color = GREEN if phase != "sampling" else BLUE
            pygame.draw.circle(self._screen, nose_color, (nose_x, nose_y), DOT_RADIUS)
            pygame.draw.circle(self._screen, WHITE, (nose_x, nose_y), DOT_RADIUS + 2, 2)
            pygame.draw.circle(self._screen, eye_color, (eye_x, eye_y), DOT_RADIUS)
            pygame.draw.circle(self._screen, WHITE, (eye_x, eye_y), DOT_RADIUS + 2, 2)
            if aligned:
                ring_radius = int(round(DOT_RADIUS * CALIB_ALIGNMENT_RING_SCALE))
                pygame.draw.circle(
                    self._screen, WHITE, (nose_x, nose_y), ring_radius, 2
                )

        if phase == "blink_pre" and not aligned:
            self._draw_alignment_arrow()

        instruction = current_point.get("instruction", "")
        label = f"{self._current_calib_idx + 1}/{len(self._calibration_sequence)}"
        text = f"{label}  {instruction}"
        text_surface = self._font.render(text, True, WHITE)
        text_rect = text_surface.get_rect(center=(self._width // 2, 40))
        self._screen.blit(text_surface, text_rect)

    def _draw_alignment_arrow(self) -> None:
        """Draw a center-screen arrow showing which direction to turn the nose."""
        err_yaw = self._alignment_error_yaw
        err_pitch = self._alignment_error_pitch
        if err_yaw is None or err_pitch is None:
            return
        magnitude_deg = math.hypot(err_yaw, err_pitch)
        if magnitude_deg < CALIB_ARROW_MIN_DEG:
            return
        dx = err_yaw * CALIB_ARROW_PIXELS_PER_DEG
        dy = -err_pitch * CALIB_ARROW_PIXELS_PER_DEG
        length_px = math.hypot(dx, dy)
        if length_px > CALIB_ARROW_MAX_PX:
            scale = CALIB_ARROW_MAX_PX / length_px
            dx *= scale
            dy *= scale
            length_px = CALIB_ARROW_MAX_PX
        start = (int(round(self._center_x)), int(round(self._center_y)))
        end = (int(round(self._center_x + dx)), int(round(self._center_y + dy)))
        pygame.draw.line(self._screen, WHITE, start, end, 4)
        heading = math.atan2(dy, dx)
        head_len = max(12.0, min(24.0, length_px * 0.3))
        wing_angle = math.radians(28.0)
        for sign in (-1, 1):
            angle = heading + math.pi - sign * wing_angle
            wing_end = (
                int(round(end[0] + head_len * math.cos(angle))),
                int(round(end[1] + head_len * math.sin(angle))),
            )
            pygame.draw.line(self._screen, WHITE, end, wing_end, 4)
        label = f"{magnitude_deg:.1f}°"
        label_surface = self._font.render(label, True, WHITE)
        label_offset = 16
        label_rect = label_surface.get_rect(
            center=(end[0], end[1] - label_offset if dy >= 0 else end[1] + label_offset)
        )
        self._screen.blit(label_surface, label_rect)

    def _make_calib_seq(self, width: float, height: float) -> List[Dict]:
        """Generate the calibration target list."""
        inset = CALIB_INSET
        w = width
        h = height
        center_x = w / 2
        center_y = h / 2
        t = (center_x, inset)
        b = (center_x, h - inset)
        l = (inset, center_y)
        r = (w - inset, center_y)
        tl = (inset, inset)
        tr = (w - inset, inset)
        br = (w - inset, h - inset)
        bl = (inset, h - inset)

        return [
            {
                "name": "C",
                "x": center_x,
                "y": center_y,
                "nose_x": center_x,
                "nose_y": center_y,
                "eye_x": center_x,
                "eye_y": center_y,
                "instruction": "Look forward. Keep head and eyes centered.",
            },
            {
                "name": "T",
                "x": b[0],
                "y": b[1],
                "nose_x": t[0],
                "nose_y": t[1],
                "eye_x": b[0],
                "eye_y": b[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "TL",
                "x": br[0],
                "y": br[1],
                "nose_x": tl[0],
                "nose_y": tl[1],
                "eye_x": br[0],
                "eye_y": br[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "L",
                "x": r[0],
                "y": r[1],
                "nose_x": l[0],
                "nose_y": l[1],
                "eye_x": r[0],
                "eye_y": r[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "BL",
                "x": tr[0],
                "y": tr[1],
                "nose_x": bl[0],
                "nose_y": bl[1],
                "eye_x": tr[0],
                "eye_y": tr[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "B",
                "x": t[0],
                "y": t[1],
                "nose_x": b[0],
                "nose_y": b[1],
                "eye_x": t[0],
                "eye_y": t[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "BR",
                "x": tl[0],
                "y": tl[1],
                "nose_x": br[0],
                "nose_y": br[1],
                "eye_x": tl[0],
                "eye_y": tl[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "R",
                "x": l[0],
                "y": l[1],
                "nose_x": r[0],
                "nose_y": r[1],
                "eye_x": l[0],
                "eye_y": l[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
            {
                "name": "TR",
                "x": bl[0],
                "y": bl[1],
                "nose_x": tr[0],
                "nose_y": tr[1],
                "eye_x": bl[0],
                "eye_y": bl[1],
                "instruction": "Point nose to RED and eyes to GREEN.",
            },
        ]

    def request_exit(self) -> None:
        """Signal the calibration window to close."""
        self._should_exit = True

    def is_running(self) -> bool:
        """Report whether calibration window should continue."""
        return self._running and not self._should_exit
