"""
FreeTrack 2.0 Enhanced output.
Lets FreeTrack and TrackIR (NPClient) games read the calibrated gaze directly, without opentrack running.
"""

import ctypes
import logging
import math
import mmap
import os
import subprocess
import winreg
from ctypes import wintypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import CalibratedFaceAndGazeEvent
from .facemesh_dao import FaceMeshEvent

logger = logging.getLogger(__name__)

FREETRACK_HEAP = "FT_SharedMem"
FREETRACK_MUTEX = "FT_Mutext"

INTERFACE_BOTH = "both"
INTERFACE_FREETRACK = "freetrack"
INTERFACE_NPCLIENT = "npclient"
INTERFACES = (INTERFACE_BOTH, INTERFACE_FREETRACK, INTERFACE_NPCLIENT)

OPENTRACK_UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{63F53541-A29E-4B53-825A-9B6F876A2BD6}_is1"
)
FREETRACK_REGISTRY_KEY = r"Software\Freetrack\FreetrackClient"
NPCLIENT_REGISTRY_KEY = r"Software\NaturalPoint\NATURALPOINT\NPClient Location"
FREETRACK_LIBRARIES = ("freetrackclient.dll", "freetrackclient64.dll")
NPCLIENT_LIBRARIES = ("NPClient.dll", "NPClient64.dll")
TRACKIR_DUMMY = "TrackIR.exe"
OPENTRACK_MODULES_DIR = "modules"
OPENTRACK_GAME_LIST = Path("doc", "settings", "facetracknoir supported games.csv")

MUTEX_TIMEOUT_MS = 16
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080


class FTData(ctypes.Structure):
    """Pose block games read through the FreeTrack client library."""

    _fields_ = [
        ("DataID", ctypes.c_uint32),
        ("CamWidth", ctypes.c_int32),
        ("CamHeight", ctypes.c_int32),
        ("Yaw", ctypes.c_float),
        ("Pitch", ctypes.c_float),
        ("Roll", ctypes.c_float),
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
        ("Z", ctypes.c_float),
        ("RawYaw", ctypes.c_float),
        ("RawPitch", ctypes.c_float),
        ("RawRoll", ctypes.c_float),
        ("RawX", ctypes.c_float),
        ("RawY", ctypes.c_float),
        ("RawZ", ctypes.c_float),
        ("X1", ctypes.c_float),
        ("Y1", ctypes.c_float),
        ("X2", ctypes.c_float),
        ("Y2", ctypes.c_float),
        ("X3", ctypes.c_float),
        ("Y3", ctypes.c_float),
        ("X4", ctypes.c_float),
        ("Y4", ctypes.c_float),
    ]


class FTHeap(ctypes.Structure):
    """Shared memory block of the FreeTrack 2.0 Enhanced interface, including the NPClient game handshake."""

    _fields_ = [
        ("data", FTData),
        ("GameID", ctypes.c_int32),
        ("table", ctypes.c_ubyte * 8),
        ("GameID2", ctypes.c_int32),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
_kernel32.ReleaseMutex.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.CloseHandle.restype = wintypes.BOOL


def _has_libraries(opentrack_dir: Path, libraries: Tuple[str, ...]) -> bool:
    modules = opentrack_dir / OPENTRACK_MODULES_DIR
    return all((modules / name).is_file() for name in libraries)


def find_opentrack_dir() -> Optional[Path]:
    """Locate the opentrack installation that ships the FreeTrack and NPClient client libraries."""
    candidates: List[Path] = []
    for hive, view in (
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_CURRENT_USER, 0),
    ):
        try:
            with winreg.OpenKey(
                hive, OPENTRACK_UNINSTALL_KEY, 0, winreg.KEY_READ | view
            ) as key:
                location, _ = winreg.QueryValueEx(key, "InstallLocation")
        except OSError:
            continue
        if location:
            candidates.append(Path(location))
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root, "opentrack"))

    for candidate in candidates:
        if _has_libraries(candidate, FREETRACK_LIBRARIES + NPCLIENT_LIBRARIES):
            return candidate
    return None


def load_game_keys(game_list: Path) -> Dict[int, Tuple[str, bytes]]:
    """Map NPClient game IDs to the game name and the data key that game expects."""
    games: Dict[int, Tuple[str, bytes]] = {}
    with game_list.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\r\n").split(";")
            if len(fields) != 8 or not fields[6].isdigit():
                continue
            name, since, ftn_id = fields[1], fields[3], fields[7]
            key = bytes(8)
            if since != "V160" and len(ftn_id) == 22:
                try:
                    raw = bytes.fromhex(ftn_id)
                except ValueError:
                    logger.warning(f"FreeTrack: malformed game key for '{name}'")
                else:
                    key = raw[5:1:-1] + raw[9:5:-1]
            games.setdefault(int(fields[6]), (name, key))
    return games


def _set_registry_path(key_path: str, location: str) -> None:
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, "Path", 0, winreg.REG_SZ, location)


class FreeTrackForwardStep:
    """Final pipeline step: Publish the calibrated gaze to games over the FreeTrack 2.0 Enhanced interface.

    Games see the same view rotation opentrack's freetrack protocol would give them for
    the pose the OpenTrack step forwards, so opentrack is not needed as an intermediary.
    """

    def __init__(
        self,
        interface: str = INTERFACE_BOTH,
        opentrack_dir: Optional[Path] = None,
        enabled: bool = False,
    ):
        """Initialize FreeTrack forward step.

        Args:
            interface: Client interface games may use: "both", "freetrack" or "npclient"
            opentrack_dir: opentrack installation providing the client libraries (auto-detected if None)
            enabled: Whether FreeTrack publishing is active (default: False)
        """
        if interface not in INTERFACES:
            raise ValueError(f"Unknown FreeTrack interface '{interface}'")
        self.interface = interface
        self.opentrack_dir = (
            opentrack_dir if opentrack_dir is not None else find_opentrack_dir()
        )
        self.enabled = False
        self._mutex = None
        self._mapping: Optional[mmap.mmap] = None
        self._heap: Optional[FTHeap] = None
        self._trackir_dummy: Optional[subprocess.Popen] = None
        self._game_keys: Dict[int, Tuple[str, bytes]] = {}
        self._game_id = -1

        self.set_enabled(enabled)

        logger.debug(
            f"FreeTrackForwardStep initialized: interface={interface}, "
            f"opentrack_dir={self.opentrack_dir}, enabled={self.enabled}"
        )

    @property
    def _uses_freetrack(self) -> bool:
        return self.interface != INTERFACE_NPCLIENT

    @property
    def _uses_npclient(self) -> bool:
        return self.interface != INTERFACE_FREETRACK

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable FreeTrack publishing.

        Args:
            enabled: Whether to enable FreeTrack publishing
        """
        if self.enabled == enabled:
            return

        if enabled:
            try:
                self._open()
            except OSError as e:
                logger.error(f"FreeTrack: failed to open shared memory: {e}")
                self.close()
                return
            self.enabled = True
        else:
            self.close()

        logger.debug(f"FreeTrackForwardStep enabled: {self.enabled}")

    def _open(self) -> None:
        mutex = _kernel32.CreateMutexW(None, False, FREETRACK_MUTEX)
        if not mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        self._mutex = mutex
        self._mapping = mmap.mmap(-1, ctypes.sizeof(FTHeap), tagname=FREETRACK_HEAP)
        self._heap = FTHeap.from_buffer(self._mapping)
        self._game_id = -1

        if not self._acquire():
            raise TimeoutError("FreeTrack shared memory is locked by another process")
        try:
            self._heap.GameID2 = 0
            ctypes.memset(self._heap.table, 0, ctypes.sizeof(self._heap.table))
        finally:
            _kernel32.ReleaseMutex(self._mutex)

        if self._register_client_libraries():
            self._load_game_keys()
            self._start_trackir_dummy()

        logger.info(
            f"FreeTrack 2.0 Enhanced output active (interface: {self.interface})"
        )

    def _register_client_libraries(self) -> bool:
        if self.opentrack_dir is None:
            logger.warning(
                "FreeTrack: no opentrack installation found; games only find the data "
                "if a FreeTrack/NPClient library location is already registered. "
                "Pass --opentrack-dir to register one."
            )
            return False
        required = (FREETRACK_LIBRARIES if self._uses_freetrack else ()) + (
            NPCLIENT_LIBRARIES if self._uses_npclient else ()
        )
        if not _has_libraries(self.opentrack_dir, required):
            logger.error(
                f"FreeTrack: {self.opentrack_dir / OPENTRACK_MODULES_DIR} does not contain "
                f"{', '.join(required)}; client library location not registered"
            )
            return False

        location = (self.opentrack_dir / OPENTRACK_MODULES_DIR).as_posix().rstrip("/") + "/"
        try:
            _set_registry_path(
                FREETRACK_REGISTRY_KEY, location if self._uses_freetrack else ""
            )
            _set_registry_path(
                NPCLIENT_REGISTRY_KEY, location if self._uses_npclient else ""
            )
        except OSError as e:
            logger.error(f"FreeTrack: failed to register client libraries: {e}")
            return False
        logger.info(f"FreeTrack: client libraries registered from {location}")
        return True

    def _load_game_keys(self) -> None:
        game_list = self.opentrack_dir / OPENTRACK_GAME_LIST
        try:
            self._game_keys = load_game_keys(game_list)
        except OSError as e:
            logger.warning(
                f"FreeTrack: cannot read game list {game_list}: {e}; "
                "TrackIR games that require a data key will not accept the pose"
            )

    def _start_trackir_dummy(self) -> None:
        if not self._uses_npclient:
            return
        dummy = self.opentrack_dir / OPENTRACK_MODULES_DIR / TRACKIR_DUMMY
        if not dummy.is_file():
            return
        try:
            self._trackir_dummy = subprocess.Popen([str(dummy)])
        except OSError as e:
            logger.warning(f"FreeTrack: failed to start {dummy}: {e}")

    def _stop_trackir_dummy(self) -> None:
        if self._trackir_dummy is None:
            return
        self._trackir_dummy.terminate()
        try:
            self._trackir_dummy.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._trackir_dummy.kill()
        self._trackir_dummy = None

    def close(self) -> None:
        """Stop publishing and release the shared memory and helper process."""
        self.enabled = False
        self._stop_trackir_dummy()
        self._heap = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._mutex is not None:
            _kernel32.CloseHandle(self._mutex)
            self._mutex = None

    def _acquire(self) -> bool:
        result = _kernel32.WaitForSingleObject(self._mutex, MUTEX_TIMEOUT_MS)
        return result in (WAIT_OBJECT_0, WAIT_ABANDONED)

    def _acknowledge_game(self, game_id: int) -> None:
        name, key = self._game_keys.get(game_id, ("", bytes(8)))
        ctypes.memmove(self._heap.table, key, len(key))
        self._heap.GameID2 = game_id
        self._game_id = game_id
        if game_id:
            logger.info(
                f"FreeTrack: game connected: {name or 'unknown game'} (id {game_id})"
            )

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
    ) -> None:
        """Publish calibrated gaze as the game view rotation.

        Args:
            frame: Input frame (not used but kept for interface consistency)
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)
        """
        if not self.enabled or calibrated_event is None:
            return

        try:
            yaw = -math.radians(float(calibrated_event.corrected_yaw))
            pitch = -math.radians(float(calibrated_event.corrected_pitch))
        except (TypeError, ValueError) as e:
            logger.debug(f"FreeTrack: skipping frame without calibrated gaze: {e}")
            return

        if not self._acquire():
            logger.debug("FreeTrack: shared memory busy, skipping frame")
            return
        try:
            data = self._heap.data
            data.Yaw = yaw
            data.Pitch = pitch
            data.RawYaw = yaw
            data.RawPitch = pitch
            game_id = self._heap.GameID
            if game_id != self._game_id:
                self._acknowledge_game(game_id)
                data.DataID = 0
            else:
                data.DataID = (data.DataID + 1) & 0xFFFFFFFF
        finally:
            _kernel32.ReleaseMutex(self._mutex)
