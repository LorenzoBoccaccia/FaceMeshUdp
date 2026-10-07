#!/usr/bin/env python3
"""
Harmonization Capture Script
Captures raw FaceMesh data for specific head and eye movements to analyze coordinate systems.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from capture_frame_flow import face_to_raw_result, finalize_ui_frame, measure_face
from facemesh_app.face_landmarker import FaceLandmarker, ensure_bundle
from facemesh_app.harmonization_contract import (
    HARMONIZATION_PROMPTS,
    HARMONIZATION_SCHEMA_VERSION,
    HARMONIZATION_TEST_CASE,
)


def safe_float(v, fallback=0.0):
    try:
        f = float(v)
    except (ValueError, TypeError):
        return fallback
    return f if math.isfinite(f) else fallback


# Constants
OUTPUT_DIR = Path("harmonization_data")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
HUD_BG = (20, 20, 20)
HUD_BORDER = (230, 230, 230)
HUD_TEXT = (245, 245, 245)

PROMPTS = HARMONIZATION_PROMPTS


def open_camera(camera_index: int = 0) -> Tuple[cv2.VideoCapture, Dict]:
    """Open camera with sensible defaults."""
    backends = [
        (cv2.CAP_MSMF, "msmf"),
        (cv2.CAP_DSHOW, "dshow"),
        (None, "any"),
    ]

    for backend, name in backends:
        cap = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not cap.isOpened():
            cap.release()
            continue

        # Set preferred settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue

        info = {
            "backend": name,
            "index": camera_index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
        print(f"Camera opened: {info}")
        return cap, info

    raise RuntimeError("Failed to open camera")


def draw_text_with_background(
    img,
    text: str,
    position: Tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    thickness=2,
    padding=10,
    bg_color=HUD_BG,
    text_color=HUD_TEXT,
):
    """Draw text with background rectangle."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding - baseline),
        (x + text_w + padding, y + padding + baseline),
        bg_color,
        -1,
    )
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding - baseline),
        (x + text_w + padding, y + padding + baseline),
        HUD_BORDER,
        2,
    )
    cv2.putText(
        img,
        text,
        (x, y + baseline),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )



@dataclass
class HarmonizationPoint:
    """Data captured for a single harmonization point."""

    name: str
    instruction: str
    movement_type: str
    movement_axis: str
    movement_direction: str
    timestamp_ms: int
    raw_result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "instruction": self.instruction,
            "movementType": self.movement_type,
            "movementAxis": self.movement_axis,
            "movementDirection": self.movement_direction,
            "timestampMs": self.timestamp_ms,
            "rawResult": self.raw_result,
        }


class HarmonizationCapture:
    """Main harmonization capture class."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.landmarker = None
        self.captured_data: List[HarmonizationPoint] = []
        self.current_prompt_index = 0
        self.running = False
        self.mouse_clicked = False
        self.mouse_pos = (0, 0)
        self.last_result = None

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def init_camera(self):
        """Initialize camera."""
        self.cap, self.camera_info = open_camera(self.camera_index)

    def init_landmarker(self):
        """Load the face landmarker."""
        self.landmarker = FaceLandmarker.from_bundle(ensure_bundle(), with_blendshapes=True)
        print("FaceLandmarker initialized")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
            self.mouse_pos = (x, y)

    def draw_ui(self, frame, prompt: Dict[str, Any], progress: int, total: int):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), HUD_BG, -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        # Progress indicator
        progress_text = f"Progress: {progress + 1}/{total}"
        draw_text_with_background(frame, progress_text, (30, 40), font_scale=0.8)

        # Main instruction
        instruction = prompt["instruction"]
        prompt_type = str(prompt.get("type") or "")
        if prompt_type == "head":
            instruction_bg = (40, 80, 40)
            type_color = GREEN
        elif prompt_type == "eye":
            instruction_bg = (80, 60, 40)
            type_color = YELLOW
        else:
            instruction_bg = (40, 40, 80)
            type_color = WHITE
        draw_text_with_background(
            frame,
            instruction,
            (30, 100),
            font_scale=1.2,
            bg_color=instruction_bg,
        )

        # Type indicator
        type_text = f"Type: {prompt_type.upper()} movement"
        cv2.putText(
            frame,
            type_text,
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            type_color,
            2,
            cv2.LINE_AA,
        )

        # Click instruction
        click_text = "CLICK anywhere or press SPACE to capture"
        cv2.putText(
            frame,
            click_text,
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        # Face detection indicator
        if self.landmarker is not None:
            has_face = self.last_result is not None
            face_text = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
            face_color = GREEN if has_face else RED
            cv2.putText(
                frame,
                face_text,
                (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                face_color,
                2,
                cv2.LINE_AA,
            )

        return frame

    def capture_point(self, prompt: Dict[str, Any]) -> Optional[HarmonizationPoint]:
        """Capture a single harmonization point."""
        self.mouse_clicked = False

        print(f"\n{'=' * 60}")
        print(f"CAPTURE {self.current_prompt_index + 1}/{len(PROMPTS)}")
        print(f"Instruction: {prompt['instruction']}")
        print(f"Type: {prompt['type']}")
        print(f"Click or press SPACE to capture...")
        print(f"{'=' * 60}\n")

        cv2.namedWindow("Harmonization Capture")
        cv2.setMouseCallback("Harmonization Capture", self.mouse_callback)

        while self.running:
            # Read frame
            ok, frame_bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            result = measure_face(self.landmarker, frame_bgr)
            self.last_result = result

            frame_with_ui = self.draw_ui(
                frame_bgr.copy(), prompt, self.current_prompt_index, len(PROMPTS)
            )
            frame_with_ui = finalize_ui_frame(frame_with_ui)

            cv2.imshow("Harmonization Capture", frame_with_ui)

            # Check for capture trigger
            key = cv2.waitKey(1) & 0xFF
            if self.mouse_clicked or key == ord(" "):
                # Capture the point
                print(f"Captured: {prompt['name']}")

                # Serialize the result
                raw_data = face_to_raw_result(result)

                # Create harmonization point
                point = HarmonizationPoint(
                    name=prompt["name"],
                    instruction=prompt["instruction"],
                    movement_type=prompt["type"],
                    movement_axis=prompt.get("axis", ""),
                    movement_direction=prompt.get("direction", ""),
                    timestamp_ms=int(time.time() * 1000),
                    raw_result=raw_data,
                )

                # Also save individual JSON file
                json_file = OUTPUT_DIR / f"{prompt['name']}.json"
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(point.to_dict(), f, indent=2)

                print(f"Saved: {json_file}")

                # Save a screenshot too
                screenshot_file = OUTPUT_DIR / f"{prompt['name']}.png"
                cv2.imwrite(str(screenshot_file), frame_with_ui)
                print(f"Screenshot saved: {screenshot_file}")

                cv2.waitKey(500)  # Brief pause to show feedback
                break

            if key == ord("q") or key == 27:  # q or ESC
                print("User cancelled")
                self.running = False
                return None

        return point

    def run(self):
        """Run the harmonization capture session."""
        print("\n" + "=" * 60)
        print("HARMONIZATION CAPTURE")
        print("=" * 60)
        print(f"\nThis script will capture FaceMesh data for {len(PROMPTS)} different")
        print("head, eye, and translation movements to help analyze coordinate systems.")
        print("\nInstructions:")
        print("- Follow each prompt to move your head or eyes")
        print("- Keep the movement steady when instructed")
        print("- Click anywhere or press SPACE to capture")
        print("- Press 'q' or ESC to quit early")
        print("\n" + "=" * 60 + "\n")

        # Initialize
        self.init_camera()
        self.init_landmarker()
        self.running = True

        try:
            # Capture each prompt
            for i, prompt in enumerate(PROMPTS):
                self.current_prompt_index = i

                if not self.running:
                    break

                point = self.capture_point(prompt)
                if point:
                    self.captured_data.append(point)

            # Save all data to a combined file
            if self.captured_data:
                combined_file = OUTPUT_DIR / "harmonization_combined.json"
                combined_data = {
                    "schemaVersion": HARMONIZATION_SCHEMA_VERSION,
                    "suite": "harmonization",
                    "timestamp": int(time.time() * 1000),
                    "captureCount": len(self.captured_data),
                    "cameraInfo": self.camera_info,
                    "prompts": PROMPTS,
                    "testCase": HARMONIZATION_TEST_CASE,
                    "points": [p.to_dict() for p in self.captured_data],
                }

                with combined_file.open("w", encoding="utf-8") as f:
                    json.dump(combined_data, f, indent=2)

                print(f"\nCombined data saved: {combined_file}")

            print(f"\n{'=' * 60}")
            print(f"Harmonization capture complete!")
            print(f"Captured {len(self.captured_data)} points")
            print(f"Data saved to: {OUTPUT_DIR.absolute()}")
            print(f"{'=' * 60}\n")

        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Harmonization capture for FaceMesh coordinate system analysis"
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index (default: 0)"
    )
    args = parser.parse_args()

    capture = HarmonizationCapture(camera_index=args.camera_index)
    capture.run()


if __name__ == "__main__":
    main()
