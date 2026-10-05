"""
Gaze markers drawn on overlays: where the user looks, and where the head and the eyes alone point.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2

from .calibration import CalibratedFaceAndGazeEvent
from .facemesh_dao import clamp

COMPONENT_HEAD = "head"
COMPONENT_EYE = "eye"
COMPONENT_GAZE = "gaze"

_RGB_BY_COMPONENT = {
    COMPONENT_HEAD: (255, 40, 40),
    COMPONENT_EYE: (70, 180, 255),
    COMPONENT_GAZE: (80, 230, 120),
}


@dataclass(frozen=True)
class GazePrimitive:
    """One marker on the screen and the gaze component it shows."""

    component: str
    point: Tuple[int, int]


def collect_gaze_primitives(
    calibrated_event: Optional[CalibratedFaceAndGazeEvent],
) -> List[GazePrimitive]:
    """Screen markers for the head, eye and combined gaze, kept on screen when they fall outside it."""
    if calibrated_event is None:
        return []
    gaze = calibrated_event.gaze
    screen = calibrated_event.model.screen
    primitives = []
    for component, point in (
        (COMPONENT_HEAD, gaze.head_screen_px),
        (COMPONENT_EYE, gaze.eye_screen_px),
        (COMPONENT_GAZE, gaze.screen_px),
    ):
        if point is None or not all(math.isfinite(v) for v in point):
            continue
        primitives.append(
            GazePrimitive(
                component,
                (
                    int(round(clamp(point[0], 0, screen.width_px - 1))),
                    int(round(clamp(point[1], 0, screen.height_px - 1))),
                ),
            )
        )
    return primitives


def draw_gaze_primitives_pygame(
    surface, primitives: Sequence[GazePrimitive], *, radius: int
) -> None:
    """Render gaze markers into a pygame surface."""
    import pygame

    for primitive in primitives:
        pygame.draw.circle(
            surface, _RGB_BY_COMPONENT[primitive.component], primitive.point, int(radius)
        )


def draw_gaze_primitives_cv2(
    image, primitives: Sequence[GazePrimitive], *, radius: int
) -> None:
    """Render gaze markers into a BGR image."""
    for primitive in primitives:
        r, g, b = _RGB_BY_COMPONENT[primitive.component]
        cv2.circle(image, primitive.point, int(radius), (b, g, r), -1, cv2.LINE_AA)
