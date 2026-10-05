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
from .pipeline_steps import GazeDirection

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
OPENTRACK_RELEASES_URL = "https://github.com/opentrack/opentrack/releases"

MAX_VIEW_YAW_DEG = 180.0
MAX_VIEW_PITCH_DEG = 90.0

MUTEX_TIMEOUT_MS = 16
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class FreeTrackSetupError(Exception):
    """FreeTrack output cannot be provided; the message tells the user how to fix it."""


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
_kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.SetInformationJobObject.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
)
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def required_libraries(interface: str) -> Tuple[str, ...]:
    """Client libraries games need on disk for the chosen interface."""
    freetrack = FREETRACK_LIBRARIES if interface != INTERFACE_NPCLIENT else ()
    npclient = NPCLIENT_LIBRARIES if interface != INTERFACE_FREETRACK else ()
    return freetrack + npclient


def missing_libraries(opentrack_dir: Path, interface: str) -> List[str]:
    """Client libraries for the chosen interface that this opentrack folder lacks."""
    modules = opentrack_dir / OPENTRACK_MODULES_DIR
    return [
        name for name in required_libraries(interface) if not (modules / name).is_file()
    ]


def _read_registry_string(hive, key_path: str, value: str, view: int = 0) -> str:
    try:
        with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:
            data, _ = winreg.QueryValueEx(key, value)
    except OSError:
        return ""
    return data if isinstance(data, str) else ""


def opentrack_dir_candidates() -> List[Path]:
    """Places an opentrack installation may live: installer records, folders opentrack
    registered for games when it last ran (covers portable copies), and Program Files."""
    found: List[Path] = []
    for hive, view in (
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_CURRENT_USER, 0),
    ):
        location = _read_registry_string(
            hive, OPENTRACK_UNINSTALL_KEY, "InstallLocation", view
        )
        if location:
            found.append(Path(location))
    for key_path in (NPCLIENT_REGISTRY_KEY, FREETRACK_REGISTRY_KEY):
        registered = _read_registry_string(winreg.HKEY_CURRENT_USER, key_path, "Path")
        if registered:
            library_dir = Path(registered.rstrip("/\\"))
            if library_dir.name.lower() == OPENTRACK_MODULES_DIR:
                found.append(library_dir.parent)
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            found.append(Path(root, "opentrack"))

    unique: List[Path] = []
    seen = set()
    for candidate in found:
        key = os.path.normcase(os.path.normpath(str(candidate)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_opentrack_dir(explicit: Optional[Path], interface: str) -> Path:
    """Find the opentrack installation whose client libraries games will load."""
    libraries = ", ".join(required_libraries(interface))
    if explicit is not None:
        missing = missing_libraries(explicit, interface)
        if missing:
            raise FreeTrackSetupError(
                f"FreeTrack output needs opentrack's client libraries, but "
                f"{explicit / OPENTRACK_MODULES_DIR} is missing {', '.join(missing)}.\n"
                f"Point --opentrack-dir (or OPENTRACK_DIR) at the opentrack installation "
                f"folder, the one containing opentrack.exe, or reinstall opentrack from "
                f"{OPENTRACK_RELEASES_URL}"
            )
        return explicit

    candidates = opentrack_dir_candidates()
    for candidate in candidates:
        if not missing_libraries(candidate, interface):
            return candidate
    looked_in = (
        "\n".join(f"  {candidate / OPENTRACK_MODULES_DIR}" for candidate in candidates)
        or "  (no opentrack installation recorded on this machine)"
    )
    raise FreeTrackSetupError(
        f"FreeTrack output needs opentrack installed: games read the pose through its "
        f"client libraries ({libraries}).\n"
        f"No opentrack installation with those files was found. Looked in:\n"
        f"{looked_in}\n"
        f"Install opentrack from {OPENTRACK_RELEASES_URL}, or point to an existing "
        f"installation with --opentrack-dir or the OPENTRACK_DIR environment variable."
    )


def _kill_with_this_process(process: subprocess.Popen):
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = _JobExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    handle = _kernel32.OpenProcess(
        PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, process.pid
    )
    bound = bool(
        handle
        and _kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        and _kernel32.AssignProcessToJobObject(job, handle)
    )
    if handle:
        _kernel32.CloseHandle(handle)
    if not bound:
        _kernel32.CloseHandle(job)
        return None
    return job


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

    Games see the view rotation opentrack's freetrack protocol would give them for the
    pose the OpenTrack step forwards, scaled by the output multiplier so a small gaze shift
    can turn the game view further. opentrack has to be installed but not running.
    """

    def __init__(
        self,
        opentrack_dir: Path,
        interface: str = INTERFACE_BOTH,
        multiplier: float = 1.0,
        enabled: bool = False,
    ):
        """Initialize FreeTrack forward step.

        Args:
            opentrack_dir: opentrack installation whose client libraries games load
            interface: Client interface games may use: "both", "freetrack" or "npclient"
            multiplier: Game view angle per degree of gaze, positive (default: 1)
            enabled: Whether FreeTrack publishing is active (default: False)

        Raises:
            FreeTrackSetupError: Enabling failed; the message explains the fix
        """
        if interface not in INTERFACES:
            raise ValueError(f"Unknown FreeTrack interface '{interface}'")
        if multiplier <= 0:
            raise ValueError(f"FreeTrack multiplier must be positive, got {multiplier}")
        self.interface = interface
        self.multiplier = multiplier
        self.opentrack_dir = opentrack_dir
        self.enabled = False
        self._mutex = None
        self._mapping: Optional[mmap.mmap] = None
        self._heap: Optional[FTHeap] = None
        self._trackir_dummy: Optional[subprocess.Popen] = None
        self._trackir_dummy_job = None
        self._game_keys: Dict[int, Tuple[str, bytes]] = {}
        self._game_id = -1

        self.set_enabled(enabled)

        logger.debug(
            f"FreeTrackForwardStep initialized: interface={interface}, "
            f"multiplier={multiplier}, opentrack_dir={self.opentrack_dir}, "
            f"enabled={self.enabled}"
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

        Raises:
            FreeTrackSetupError: Enabling failed; the message explains the fix
        """
        if self.enabled == enabled:
            return

        if enabled:
            try:
                self._open()
            except BaseException:
                self.close()
                raise
            self.enabled = True
        else:
            self.close()

        logger.debug(f"FreeTrackForwardStep enabled: {self.enabled}")

    def _open(self) -> None:
        missing = missing_libraries(self.opentrack_dir, self.interface)
        if missing:
            raise FreeTrackSetupError(
                f"FreeTrack output needs opentrack's client libraries, but "
                f"{self.opentrack_dir / OPENTRACK_MODULES_DIR} is missing "
                f"{', '.join(missing)}.\n"
                f"Reinstall opentrack from {OPENTRACK_RELEASES_URL}"
            )
        try:
            mutex = _kernel32.CreateMutexW(None, False, FREETRACK_MUTEX)
            if not mutex:
                raise ctypes.WinError(ctypes.get_last_error())
            self._mutex = mutex
            self._mapping = mmap.mmap(
                -1, ctypes.sizeof(FTHeap), tagname=FREETRACK_HEAP
            )
            self._heap = FTHeap.from_buffer(self._mapping)
        except OSError as e:
            raise FreeTrackSetupError(
                f"FreeTrack output could not create the shared memory games read: {e}"
            ) from e
        self._game_id = -1

        if not self._acquire():
            raise FreeTrackSetupError(
                "FreeTrack shared memory is held by another program. Close opentrack "
                "or any other head tracker using the freetrack protocol and try again."
            )
        try:
            self._heap.GameID2 = 0
            ctypes.memset(self._heap.table, 0, ctypes.sizeof(self._heap.table))
        finally:
            _kernel32.ReleaseMutex(self._mutex)

        self._register_client_libraries()
        if self._uses_npclient:
            self._load_game_keys()
            self._start_trackir_dummy()

        logger.info(
            f"FreeTrack 2.0 Enhanced output active (interface: {self.interface}, "
            f"multiplier: {self.multiplier:g})"
        )

    def _register_client_libraries(self) -> None:
        location = (self.opentrack_dir / OPENTRACK_MODULES_DIR).as_posix().rstrip("/") + "/"
        try:
            _set_registry_path(
                FREETRACK_REGISTRY_KEY, location if self._uses_freetrack else ""
            )
            _set_registry_path(
                NPCLIENT_REGISTRY_KEY, location if self._uses_npclient else ""
            )
        except OSError as e:
            raise FreeTrackSetupError(
                f"FreeTrack output could not tell games where opentrack's client "
                f"libraries are: writing the FreeTrack/TrackIR location under "
                f"HKEY_CURRENT_USER failed ({e})."
            ) from e
        logger.info(f"FreeTrack: games will load opentrack's client libraries from {location}")

    def _load_game_keys(self) -> None:
        game_list = self.opentrack_dir / OPENTRACK_GAME_LIST
        try:
            self._game_keys = load_game_keys(game_list)
        except OSError as e:
            logger.warning(
                f"FreeTrack: cannot read opentrack's game list {game_list} "
                f"({e.strerror or e}). "
                "TrackIR games that require encrypted data, such as DCS or Elite "
                f"Dangerous, will ignore the pose. Reinstall opentrack from "
                f"{OPENTRACK_RELEASES_URL} to restore it."
            )

    def _start_trackir_dummy(self) -> None:
        dummy = self.opentrack_dir / OPENTRACK_MODULES_DIR / TRACKIR_DUMMY
        if not dummy.is_file():
            logger.warning(
                f"FreeTrack: {dummy} not found. Games that wait for a running TrackIR "
                f"program will not start head tracking. Reinstall opentrack from "
                f"{OPENTRACK_RELEASES_URL} to restore it."
            )
            return
        try:
            self._trackir_dummy = subprocess.Popen([str(dummy)])
        except OSError as e:
            logger.warning(
                f"FreeTrack: failed to start {dummy} ({e}). Games that wait for a "
                "running TrackIR program will not start head tracking."
            )
            return
        self._trackir_dummy_job = _kill_with_this_process(self._trackir_dummy)
        if self._trackir_dummy_job is None:
            logger.warning(
                f"FreeTrack: {TRACKIR_DUMMY} may keep running if this app is closed "
                "abruptly; end it from Task Manager if so."
            )

    def _stop_trackir_dummy(self) -> None:
        if self._trackir_dummy is not None:
            self._trackir_dummy.terminate()
            try:
                self._trackir_dummy.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._trackir_dummy.kill()
            self._trackir_dummy = None
        if self._trackir_dummy_job is not None:
            _kernel32.CloseHandle(self._trackir_dummy_job)
            self._trackir_dummy_job = None

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

    def _view_angle(self, gaze_deg: float, limit_deg: float) -> float:
        """Game view angle for a gaze angle, kept within the range games accept."""
        return max(-limit_deg, min(limit_deg, gaze_deg * self.multiplier))

    def receive_frame(
        self,
        frame: np.ndarray,
        face_mesh_event: Optional[FaceMeshEvent],
        calibrated_event: Optional[CalibratedFaceAndGazeEvent],
        gaze: Optional[GazeDirection],
    ) -> None:
        """Publish calibrated gaze as the game view rotation.

        Args:
            frame: Input frame (not used but kept for interface consistency)
            face_mesh_event: Face mesh data (optional)
            calibrated_event: Calibrated face and gaze data (optional)
            gaze: Gaze direction to publish (optional)
        """
        if not self.enabled or gaze is None:
            return

        yaw = -math.radians(self._view_angle(gaze.yaw, MAX_VIEW_YAW_DEG))
        pitch = -math.radians(self._view_angle(gaze.pitch, MAX_VIEW_PITCH_DEG))

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
