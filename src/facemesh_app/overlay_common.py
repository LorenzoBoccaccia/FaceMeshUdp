"""
Shared overlay helpers and display/window configuration.
"""

import ctypes
import sys
from typing import Dict

import pygame


KEY_COLOR = (1, 0, 1)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 180, 255)
HUD_BG = (20, 20, 20)
RED = (255, 40, 40)
GREEN = (80, 230, 120)
DOT_RADIUS = 14


def set_window_transparent(hwnd):
    """Set the overlay window to color-key transparency."""
    if sys.platform != "win32" or not hwnd:
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    LWA_COLORKEY = 0x00000001

    colorref = KEY_COLOR[0] | (KEY_COLOR[1] << 8) | (KEY_COLOR[2] << 16)
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED)
    user32.SetLayeredWindowAttributes(hwnd, colorref, 0, LWA_COLORKEY)


def set_window_topmost(hwnd):
    """Set the overlay window as always-on-top."""
    if sys.platform != "win32" or not hwnd:
        return
    user32 = ctypes.windll.user32
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040
    SWP_NOACTIVATE = 0x0010
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW | SWP_NOACTIVATE,
    )


def set_window_click_through(hwnd):
    """Set the overlay window to ignore mouse events."""
    if sys.platform != "win32" or not hwnd:
        return
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TRANSPARENT)


def get_display_geo() -> Dict:
    """Return primary display geometry."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return {
            "name": "Primary Display",
            "x": 0,
            "y": 0,
            "width": int(user32.GetSystemMetrics(0)),
            "height": int(user32.GetSystemMetrics(1)),
        }
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if sizes:
        w, h = sizes[0]
    else:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    return {"name": "Display", "x": 0, "y": 0, "width": int(w), "height": int(h)}
