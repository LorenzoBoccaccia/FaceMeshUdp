"""
Calibration overlay window for 9-point calibration flow.
"""

import time
from typing import Dict, List, Optional, Tuple

import pyglet

from .calibration import REFERENCE_POINT, CalibrationPoint
from .facemesh_dao import FaceMeshEvent
from .overlay_common import (
    BLACK,
    BLUE,
    DOT_RADIUS,
    GREEN,
    RED,
    WHITE,
    OverlayWindow,
)


CALIB_INSET = 50
CALIB_BLINK_MS = 500
CALIB_CAPTURE_MS = 500
CALIB_REFERENCE_CAPTURE_MS = 5000
CALIB_BLINK_PERIOD_MS = 120
CALIB_FONT_SIZE = 24


class _Target:
    """A calibration dot with a white rim, hidden while it has no colour."""

    def __init__(self, batch: pyglet.graphics.Batch, rim_group, dot_group):
        self._rim = pyglet.shapes.Circle(
            0, 0, DOT_RADIUS + 2, color=WHITE, batch=batch, group=rim_group
        )
        self._dot = pyglet.shapes.Circle(0, 0, DOT_RADIUS, batch=batch, group=dot_group)
        self.show(None, None)

    def show(
        self,
        position: Optional[Tuple[float, float]],
        color: Optional[Tuple[int, int, int]],
    ) -> None:
        """Place the target in the given colour, or hide it when it has no position or colour."""
        visible = position is not None and color is not None
        self._rim.visible = visible
        self._dot.visible = visible
        if visible:
            self._rim.position = position
            self._dot.position = position
            self._dot.color = color


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

        self._window: Optional[OverlayWindow] = None
        self._nose_target: Optional[_Target] = None
        self._eye_target: Optional[_Target] = None
        self._text: Optional[pyglet.text.Label] = None

        self._calibration_sequence: List[Dict] = []
        self._current_calib_idx: int = 0
        self._calib_phase: str = "idle"
        self._calib_phase_start: int = 0
        self._calib_samples: List[FaceMeshEvent] = []
        self._retry_point = False
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._click_pending: bool = False

    def initialize(self):
        """Create and configure calibration overlay window."""
        self._window = OverlayWindow(
            self._display,
            "FaceMesh Calibration",
            self._overlay_fps,
            click_through=False,
            background=BLACK,
        )
        batch = self._window.batch
        rim_group = pyglet.graphics.Group(order=0)
        dot_group = pyglet.graphics.Group(order=1)
        self._nose_target = _Target(batch, rim_group, dot_group)
        self._eye_target = _Target(batch, rim_group, dot_group)
        self._text = pyglet.text.Label(
            "",
            x=self._width / 2,
            y=self._height / 4,
            width=self._width,
            multiline=True,
            align="center",
            anchor_x="center",
            anchor_y="top",
            font_size=CALIB_FONT_SIZE,
            color=(*WHITE, 255),
            batch=batch,
            group=pyglet.graphics.Group(order=2),
        )

    def shutdown(self):
        """Close calibration overlay resources."""
        if self._window is not None:
            self._window.close()
            self._window = None

    def handle_events(self):
        """Process calibration window events."""
        self._window.poll()
        if self._window.take_click():
            self._click_pending = True

    def render(self):
        """Render one calibration frame."""
        current_point = (
            self.get_current_calib_point() if self._calib_phase != "idle" else None
        )
        if current_point is None:
            self._nose_target.show(None, None)
            self._eye_target.show(None, None)
            self._set_text("")
        else:
            elapsed_ms = int(time.time() * 1000) - self._calib_phase_start
            self._show_calibration(current_point, self._calib_phase, elapsed_ms)
        self._window.present()

    def start_calibration_sequence(self, width: float, height: float):
        """Initialize calibration sequence targets and reset state."""
        self._calibration_sequence = self._make_calib_seq(width, height)
        self._current_calib_idx = 0
        self._calib_phase = "wait_click"
        self._calib_phase_start = int(time.time() * 1000)
        self._calib_samples = []
        self._center_x = float(width) / 2.0
        self._center_y = float(height) / 2.0
        self._click_pending = False

    def get_current_calib_point(self) -> Optional[Dict]:
        """Return current calibration target."""
        if 0 <= self._current_calib_idx < len(self._calibration_sequence):
            return self._calibration_sequence[self._current_calib_idx]
        return None

    def get_calibration_phase(self) -> str:
        """Return current calibration phase."""
        return self._calib_phase

    def is_aligned(self) -> bool:
        """Report whether the user has committed to the current target."""
        return self._calib_phase in ("blink", "capture")

    def update_calibration_state(
        self, evt: Optional[FaceMeshEvent]
    ) -> Tuple[bool, Optional[CalibrationPoint]]:
        """Advance calibration and emit a point once its capture window held usable face frames."""
        current_time = int(time.time() * 1000)
        elapsed_ms = current_time - self._calib_phase_start
        current_point = self.get_current_calib_point()
        if current_point is None:
            return True, None

        if self._calib_phase == "wait_click":
            if self._click_pending:
                self._enter_phase("blink", current_time)
        elif self._calib_phase == "blink":
            self._click_pending = False
            if elapsed_ms >= CALIB_BLINK_MS:
                self._enter_phase("capture", current_time)
        elif self._calib_phase == "capture":
            if evt is not None:
                self._calib_samples.append(evt)
            capture_ms = (
                CALIB_REFERENCE_CAPTURE_MS
                if current_point["name"] == REFERENCE_POINT
                else CALIB_CAPTURE_MS
            )
            if elapsed_ms >= capture_ms:
                calib_point = CalibrationPoint.from_samples(
                    current_point["name"],
                    (current_point["nose_x"], current_point["nose_y"]),
                    (current_point["eye_x"], current_point["eye_y"]),
                    self._calib_samples,
                )
                self._retry_point = calib_point is None
                if calib_point is not None:
                    self._current_calib_idx += 1
                self._enter_phase("wait_click", current_time)
                return self.get_current_calib_point() is None, calib_point

        return False, None

    def _enter_phase(self, phase: str, start_ms: int) -> None:
        self._calib_phase = phase
        self._calib_phase_start = start_ms
        self._calib_samples = []
        self._click_pending = False

    def _set_text(self, text: str) -> None:
        if self._text.text != text:
            self._text.text = text

    def _show_calibration(
        self,
        current_point: Dict,
        phase: str,
        elapsed_ms: int,
    ):
        nose = (current_point["nose_x"], current_point["nose_y"])
        eye = (current_point["eye_x"], current_point["eye_y"])

        if phase == "blink":
            show = (elapsed_ms // CALIB_BLINK_PERIOD_MS) % 2 == 0
            nose_color = WHITE if show else None
            eye_color = WHITE if show else None
        elif phase == "capture":
            nose_color = BLUE
            eye_color = BLUE
        else:
            nose_color = RED
            eye_color = GREEN

        self._nose_target.show(self._window.window_point(*nose), nose_color)
        self._eye_target.show(
            self._window.window_point(*eye) if eye != nose else None, eye_color
        )

        instruction = current_point.get("instruction", "")
        label = f"Step {self._current_calib_idx + 1}/{len(self._calibration_sequence)}"
        phase_hint = {
            "wait_click": (
                "No face seen — align and CLICK again."
                if self._retry_point
                else "Align, hold steady, then CLICK."
            ),
            "blink": "Hold still — capturing…",
            "capture": "Keep holding — sampling…",
        }.get(phase, "")

        lines = [label]
        lines.extend(line for line in instruction.split("\n") if line)
        if phase_hint:
            lines.append(phase_hint)
        self._set_text("\n".join(lines))

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

        def toward(point):
            return ((center_x + point[0]) / 2, (center_y + point[1]) / 2)

        click_instruction = (
            "Turn your HEAD so your nose points at the RED dot.\n"
            "Keep your EYES fixed on the GREEN dot (do not move your head to follow it)."
        )

        return [
            {
                "name": "C",
                "nose_x": center_x,
                "nose_y": center_y,
                "eye_x": center_x,
                "eye_y": center_y,
                "instruction": (
                    "Face the screen squarely and look at the CENTRE dot.\n"
                    "Keep your head and eyes aligned straight ahead; this one is held for 5 seconds."
                ),
            },
            {
                "name": "T",
                "nose_x": toward(t)[0],
                "nose_y": toward(t)[1],
                "eye_x": b[0],
                "eye_y": b[1],
                "instruction": click_instruction,
            },
            {
                "name": "TL",
                "nose_x": toward(tl)[0],
                "nose_y": toward(tl)[1],
                "eye_x": br[0],
                "eye_y": br[1],
                "instruction": click_instruction,
            },
            {
                "name": "L",
                "nose_x": toward(l)[0],
                "nose_y": toward(l)[1],
                "eye_x": r[0],
                "eye_y": r[1],
                "instruction": click_instruction,
            },
            {
                "name": "BL",
                "nose_x": toward(bl)[0],
                "nose_y": toward(bl)[1],
                "eye_x": tr[0],
                "eye_y": tr[1],
                "instruction": click_instruction,
            },
            {
                "name": "B",
                "nose_x": toward(b)[0],
                "nose_y": toward(b)[1],
                "eye_x": t[0],
                "eye_y": t[1],
                "instruction": click_instruction,
            },
            {
                "name": "BR",
                "nose_x": toward(br)[0],
                "nose_y": toward(br)[1],
                "eye_x": tl[0],
                "eye_y": tl[1],
                "instruction": click_instruction,
            },
            {
                "name": "R",
                "nose_x": toward(r)[0],
                "nose_y": toward(r)[1],
                "eye_x": l[0],
                "eye_y": l[1],
                "instruction": click_instruction,
            },
            {
                "name": "TR",
                "nose_x": toward(tr)[0],
                "nose_y": toward(tr)[1],
                "eye_x": bl[0],
                "eye_y": bl[1],
                "instruction": click_instruction,
            },
        ]

    def is_running(self) -> bool:
        """Report whether calibration window should continue."""
        return self._window is not None and not self._window.exit_requested
