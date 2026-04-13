#!/usr/bin/env python3
"""
Pitch Correlation Capture Script
Captures raw FaceMesh data for head and eye pitch combinations to analyze eye pitch consistency.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def safe_float(v, fallback=0.0):
    """Safely convert value to float with fallback."""
    try:
        f = float(v)
    except Exception:
        return fallback
    return f if math.isfinite(f) else fallback

# Constants
MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OUTPUT_DIR = Path("pitch_correlation")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
HUD_BG = (20, 20, 20)
HUD_BORDER = (230, 230, 230)
HUD_TEXT = (245, 245, 245)

# 9 prompts to capture: 3 head positions x 3 eye positions
# Head positions: down, center, up
# Eye positions: down, center, up
PROMPTS = [
    # Head DOWN row
    {"name": "head-down-eye-down", "instruction": "Tilt your HEAD DOWN and look with your EYES DOWN", "type": "combined"},
    {"name": "head-down-eye-center", "instruction": "Tilt your HEAD DOWN and look with your EYES STRAIGHT AHEAD", "type": "combined"},
    {"name": "head-down-eye-up", "instruction": "Tilt your HEAD DOWN and look with your EYES UP", "type": "combined"},
    # Head CENTER row
    {"name": "head-center-eye-down", "instruction": "Keep your HEAD CENTERED and look with your EYES DOWN", "type": "combined"},
    {"name": "head-center-eye-center", "instruction": "Keep your HEAD CENTERED and look with your EYES STRAIGHT AHEAD", "type": "combined"},
    {"name": "head-center-eye-up", "instruction": "Keep your HEAD CENTERED and look with your EYES UP", "type": "combined"},
    # Head UP row
    {"name": "head-up-eye-down", "instruction": "Tilt your HEAD UP and look with your EYES DOWN", "type": "combined"},
    {"name": "head-up-eye-center", "instruction": "Tilt your HEAD UP and look with your EYES STRAIGHT AHEAD", "type": "combined"},
    {"name": "head-up-eye-up", "instruction": "Tilt your HEAD UP and look with your EYES UP", "type": "combined"},
]


def ensure_model():
    """Download MediaPipe model if not present."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print("Model downloaded.")


def open_camera(camera_index: int = 0) -> Tuple[cv2.VideoCapture, Dict]:
    """Open camera with sensible defaults."""
    backends = [
        (cv2.CAP_MSMF, "msmf"),
        (cv2.CAP_DSHOW, "dshow"),
        (None, "any"),
    ]
    
    for backend, name in backends:
        cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
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


def draw_text_with_background(img, text: str, position: Tuple[int, int], 
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0,
                            thickness=2, padding=10, bg_color=HUD_BG, text_color=HUD_TEXT):
    """Draw text with background rectangle."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - padding, y - text_h - padding - baseline), 
                 (x + text_w + padding, y + padding + baseline), bg_color, -1)
    cv2.rectangle(img, (x - padding, y - text_h - padding - baseline), 
                 (x + text_w + padding, y + padding + baseline), HUD_BORDER, 2)
    cv2.putText(img, text, (x, y + baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)


def serialize_mediapipe_result(result) -> Dict[str, Any]:
    """Serialize MediaPipe FaceLandmarker result to JSON-compatible dict."""
    if result is None:
        return {}
    
    def serialize_landmarks(landmarks):
        if landmarks is None:
            return None
        try:
            # MediaPipe face_landmarks is a list of NormalizedLandmarkList
            if hasattr(landmarks, '__iter__') and not isinstance(landmarks, (str, bytes)):
                result = []
                for lm in landmarks:
                    # Each landmark should have x, y, z properties
                    # Use safe_float to handle None values
                    x_val = getattr(lm, 'x', None) if hasattr(lm, 'x') else None
                    y_val = getattr(lm, 'y', None) if hasattr(lm, 'y') else None
                    z_val = getattr(lm, 'z', None) if hasattr(lm, 'z') else None
                    
                    lm_data = {
                        "x": safe_float(x_val) if x_val is not None else None,
                        "y": safe_float(y_val) if y_val is not None else None,
                        "z": safe_float(z_val) if z_val is not None else None,
                    }
                    # Optional fields that may not exist in face landmarks
                    if hasattr(lm, 'visibility'):
                        v_val = getattr(lm, 'visibility', None)
                        lm_data["visibility"] = safe_float(v_val) if v_val is not None else None
                    if hasattr(lm, 'presence'):
                        p_val = getattr(lm, 'presence', None)
                        lm_data["presence"] = safe_float(p_val) if p_val is not None else None
                    result.append(lm_data)
                return result
        except Exception as e:
            print(f"Error serializing landmarks: {e}")
        return None
    
    def serialize_matrix(matrix):
        if matrix is None:
            return None
        try:
            if hasattr(matrix, 'flatten'):
                return [float(x) for x in matrix.flatten()]
            elif hasattr(matrix, '__iter__'):
                flat = []
                for row in matrix:
                    if hasattr(row, '__iter__'):
                        flat.extend([float(x) for x in row])
                    else:
                        flat.append(float(row))
                return flat
        except Exception:
            pass
        return None
    
    def serialize_blendshapes(blendshapes):
        if blendshapes is None:
            return None
        try:
            if hasattr(blendshapes, '__iter__') and not isinstance(blendshapes, (str, bytes)):
                return [
                    {
                        "category": str(bs.category) if hasattr(bs, 'category') else None,
                        "score": float(bs.score) if hasattr(bs, 'score') else None,
                    }
                    for bs in blendshapes
                ]
        except Exception:
            pass
        return None
    
    data = {}
    
    # Get facial transformation matrixes
    if hasattr(result, 'facial_transformation_matrixes'):
        fts = result.facial_transformation_matrixes
        if fts and len(fts) > 0:
            data['facial_transformation_matrix'] = serialize_matrix(fts[0])
    
    # Get face landmarks
    if hasattr(result, 'face_landmarks'):
        fl = result.face_landmarks
        if fl and len(fl) > 0:
            # MediaPipe returns a list, one per face detected
            data['face_landmarks'] = serialize_landmarks(fl[0])
        else:
            print(f"Warning: No face landmarks found in result")
    
    # Get face blendshapes
    if hasattr(result, 'face_blendshapes'):
        fbs = result.face_blendshapes
        if fbs and len(fbs) > 0:
            data['face_blendshapes'] = serialize_blendshapes(fbs[0])
    
    return data


@dataclass
class PitchCorrelationPoint:
    """Data captured for a single pitch correlation point."""
    name: str
    instruction: str
    head_position: str  # 'down', 'center', 'up'
    eye_position: str  # 'down', 'center', 'up'
    timestamp_ms: int
    raw_result: Dict[str, Any]  # Serialized MediaPipe result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "instruction": self.instruction,
            "headPosition": self.head_position,
            "eyePosition": self.eye_position,
            "timestampMs": self.timestamp_ms,
            "rawResult": self.raw_result,
        }


class PitchCorrelationCapture:
    """Main pitch correlation capture class."""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.landmarker = None
        self.captured_data: List[PitchCorrelationPoint] = []
        self.current_prompt_index = 0
        self.running = False
        self.mouse_clicked = False
        self.mouse_pos = (0, 0)
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def init_camera(self):
        """Initialize camera."""
        self.cap, self.camera_info = open_camera(self.camera_index)
    
    def init_landmarker(self):
        """Initialize MediaPipe FaceLandmarker."""
        ensure_model()
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
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
        cv2.rectangle(overlay, (0, 0), (w, 220), HUD_BG, -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        
        # Progress indicator
        progress_text = f"Progress: {progress + 1}/{total}"
        draw_text_with_background(frame, progress_text, (30, 40), font_scale=0.8)
        
        # Parse head and eye positions from name
        name_parts = prompt["name"].split("-")
        head_pos = name_parts[1].upper()
        eye_pos = name_parts[3].upper()
        
        # Position indicators
        pos_text = f"Head: {head_pos} | Eyes: {eye_pos}"
        cv2.putText(frame, pos_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
        
        # Main instruction
        instruction = prompt["instruction"]
        draw_text_with_background(frame, instruction, (30, 115), font_scale=1.0, 
                                 bg_color=(60, 60, 80))
        
        # Type indicator
        type_text = "PITCH CORRELATION TEST"
        cv2.putText(frame, type_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
        
        # Click instruction
        click_text = "CLICK anywhere or press SPACE to capture"
        cv2.putText(frame, click_text, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        
        # Face detection indicator
        if hasattr(self, 'last_result') and self.last_result:
            has_face = bool(self.last_result.face_landmarks and len(self.last_result.face_landmarks) > 0)
            face_text = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
            face_color = GREEN if has_face else RED
            cv2.putText(frame, face_text, (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2, cv2.LINE_AA)
        
        return frame
    
    def capture_point(self, prompt: Dict[str, Any]) -> Optional[PitchCorrelationPoint]:
        """Capture a single pitch correlation point."""
        self.mouse_clicked = False
        
        # Parse positions from name
        name_parts = prompt["name"].split("-")
        head_position = name_parts[1]
        eye_position = name_parts[3]
        
        print(f"\n{'='*60}")
        print(f"CAPTURE {self.current_prompt_index + 1}/{len(PROMPTS)}")
        print(f"Instruction: {prompt['instruction']}")
        print(f"Head position: {head_position}")
        print(f"Eye position: {eye_position}")
        print(f"Click or press SPACE to capture...")
        print(f"{'='*60}\n")
        
        cv2.namedWindow("Pitch Correlation Capture")
        cv2.setMouseCallback("Pitch Correlation Capture", self.mouse_callback)
        
        while self.running:
            # Read frame
            ok, frame_bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            
            # Mirror the frame horizontally (like a mirror)
            frame_bgr = cv2.flip(frame_bgr, 1)
            
            # Detect face mesh
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.landmarker.detect(mp_image)
            self.last_result = result
            
            # Draw UI
            frame_with_ui = self.draw_ui(frame_bgr.copy(), prompt, self.current_prompt_index, len(PROMPTS))
            
            # Display
            cv2.imshow("Pitch Correlation Capture", frame_with_ui)
            
            # Check for capture trigger
            key = cv2.waitKey(1) & 0xFF
            if self.mouse_clicked or key == ord(' '):
                # Capture the point
                print(f"Captured: {prompt['name']}")
                
                # Serialize the result
                raw_data = serialize_mediapipe_result(result)
                
                # Create pitch correlation point
                point = PitchCorrelationPoint(
                    name=prompt["name"],
                    instruction=prompt["instruction"],
                    head_position=head_position,
                    eye_position=eye_position,
                    timestamp_ms=int(time.time() * 1000),
                    raw_result=raw_data,
                )
                
                # Also save individual JSON file
                json_file = OUTPUT_DIR / f"{prompt['name']}.json"
                with json_file.open('w', encoding='utf-8') as f:
                    json.dump(point.to_dict(), f, indent=2)
                
                print(f"Saved: {json_file}")
                
                # Save a screenshot too
                screenshot_file = OUTPUT_DIR / f"{prompt['name']}.png"
                cv2.imwrite(str(screenshot_file), frame_with_ui)
                print(f"Screenshot saved: {screenshot_file}")
                
                cv2.waitKey(500)  # Brief pause to show feedback
                break
            
            if key == ord('q') or key == 27:  # q or ESC
                print("User cancelled")
                self.running = False
                return None
        
        return point
    
    def run(self):
        """Run the pitch correlation capture session."""
        print("\n" + "="*60)
        print("PITCH CORRELATION CAPTURE")
        print("="*60)
        print("\nThis script will capture FaceMesh data for 9 different")
        print("combinations of head and eye positions to analyze")
        print("eye pitch consistency across head positions.")
        print("\nTest pattern:")
        print("  3 head positions (DOWN, CENTER, UP)")
        print("  x 3 eye positions (DOWN, CENTER, UP)")
        print("  = 9 capture points")
        print("\nInstructions:")
        print("- Follow each prompt carefully")
        print("- Keep the position steady when instructed")
        print("- Click anywhere or press SPACE to capture")
        print("- Press 'q' or ESC to quit early")
        print("\n" + "="*60 + "\n")
        
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
                combined_file = OUTPUT_DIR / "pitch_correlation_combined.json"
                combined_data = {
                    "timestamp": int(time.time() * 1000),
                    "captureCount": len(self.captured_data),
                    "cameraInfo": self.camera_info,
                    "points": [p.to_dict() for p in self.captured_data],
                }
                
                with combined_file.open('w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2)
                
                print(f"\nCombined data saved: {combined_file}")
            
            print(f"\n{'='*60}")
            print(f"Pitch correlation capture complete!")
            print(f"Captured {len(self.captured_data)} points")
            print(f"Data saved to: {OUTPUT_DIR.absolute()}")
            print(f"{'='*60}\n")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()
            if self.landmarker is not None:
                self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pitch correlation capture for FaceMesh eye pitch consistency analysis"
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Camera index (default: 0)"
    )
    args = parser.parse_args()
    
    capture = PitchCorrelationCapture(camera_index=args.camera_index)
    capture.run()


if __name__ == "__main__":
    main()
