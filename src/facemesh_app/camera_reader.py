# camera_reader.py
"""
CameraReader module for continuous camera frame capture.
Handles camera initialization and frame dispatching to FrameDispatcher.
"""

import time
import threading
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from typing import Optional
    from .frame_dispatcher import FrameDispatcher


class CameraReader:
    """
    Continuously receives frames from camera and sends them to FrameDispatcher.
    """
    
    def __init__(
        self,
        frame_dispatcher: 'FrameDispatcher',
        camera_id: int = 0,
        backend: int = cv2.CAP_ANY,
        camera_fourcc: str = "",
        camera_width: int = 0,
        camera_height: int = 0,
        camera_fps: int = 0
    ):
        """
        Initialize CameraReader with camera configuration.
        
        Args:
            frame_dispatcher: FrameDispatcher instance to receive captured frames
            camera_id: Camera device ID (default: 0)
            backend: OpenCV backend preference (default: cv2.CAP_ANY)
            camera_fourcc: Optional fourcc codec string (e.g., "MJPG")
            camera_width: Optional camera width (0 = use default)
            camera_height: Optional camera height (0 = use default)
            camera_fps: Optional camera FPS (0 = use default)
        """
        self.frame_dispatcher = frame_dispatcher
        self.camera_id = camera_id
        self.backend = backend
        self.camera_fourcc = camera_fourcc
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None
    
    def _ms_now(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)
    
    def _open_camera(self) -> tuple[cv2.VideoCapture, dict]:
        """
        Open camera with backend fallback and configuration.
        
        Returns:
            Tuple of (VideoCapture object, camera_info dict)
            
        Raises:
            RuntimeError: If camera cannot be opened
        """
        backends = {
            "auto": [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", None)],
            "msmf": [("msmf", cv2.CAP_MSMF), ("any", None)],
            "dshow": [("dshow", cv2.CAP_DSHOW), ("any", None)],
            "any": [("any", None)],
        }
        
        # Determine which backend strategy to use based on provided backend
        backend_strategy = "any"
        if self.backend == cv2.CAP_MSMF:
            backend_strategy = "msmf"
        elif self.backend == cv2.CAP_DSHOW:
            backend_strategy = "dshow"
        
        errors = []
        
        for name, backend_value in backends.get(backend_strategy, backends["auto"]):
            cap = cv2.VideoCapture(self.camera_id, backend_value) if backend_value is not None else cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: open failed")
                continue

            fourcc = (self.camera_fourcc or "").strip().upper()
            if len(fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if self.camera_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.camera_width))
            if self.camera_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.camera_height))
            if self.camera_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(self.camera_fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, _ = cap.read()
            if not ok:
                cap.release()
                errors.append(f"{name}: read failed")
                continue

            info = {
                "backend": name,
                "index": self.camera_id,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            }
            return cap, info

        raise RuntimeError(f"Unable to open webcam ({'; '.join(errors)})")
    
    def _capture_loop(self):
        """
        Main capture loop that reads frames and sends to FrameDispatcher.
        Runs in background thread until stop event is set.
        """
        try:
            self.cap, info = self._open_camera()
            print(f"CameraReader: Camera backend={info['backend']} index={info['index']} "
                  f"{info['width']}x{info['height']} {info['fps']:.1f}fps", flush=True)
            print("CameraReader: Webcam capture started.", flush=True)
            
            while not self.stop_evt.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                
                timestamp_ms = self._ms_now()
                self.frame_dispatcher.receiveFrame(frame, timestamp_ms)
                
        except Exception as e:
            print(f"CameraReader: Capture loop failed - {e}", flush=True)
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print("CameraReader: Capture loop stopped.", flush=True)
    
    def startReceiving(self) -> None:
        """
        Start receiving frames from camera in a background thread.
        Continuously receives frames and sends them to FrameDispatcher via receiveFrame().
        """
        if self.thread is not None and self.thread.is_alive():
            print("CameraReader: Already receiving frames.", flush=True)
            return
        
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stopReceiving(self) -> None:
        """
        Stop receiving frames from camera.
        Signals the capture loop to stop and waits for thread completion.
        """
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
