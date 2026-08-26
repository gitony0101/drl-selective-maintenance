#!/usr/bin/env python3
"""Shared Heavy MPS Lock Utility for M7 and M8.

Provides mutual exclusion for heavy MPS workloads across milestones using fcntl.flock.
"""

from __future__ import annotations

import fcntl
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

# Establish PROJECT_ROOT before any local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Frozen lock path per protocol
_LOCK_PATH = Path("/tmp/drl_heavy_mps.lock")


class MPSHeavyLock:
    """Shared exclusive lock for heavy MPS operations.

    Uses fcntl.flock with LOCK_EX | LOCK_NB for non-blocking acquisition.
    Writes holder metadata while lock is held.
    """

    def __init__(
        self,
        milestone: str,
        worktree: str,
        command: str,
    ):
        """Initialize lock with holder metadata.

        Args:
            milestone: Milestone identifier (e.g., "M7", "M8")
            worktree: Git worktree path
            command: Full command being executed
        """
        self.milestone = milestone
        self.worktree = worktree
        self.command = command
        self._fd: Optional[int] = None
        self._lock_path = _LOCK_PATH
        self._pid = os.getpid()
        self._hostname = socket.gethostname()
        self._start_ts = time.time()

    def acquire(self) -> bool:
        """Attempt to acquire the lock non-blocking.

        Returns:
            True if acquired, False if contention (another process holds it)

        Raises:
            RuntimeError: If lock file operations fail unexpectedly
        """
        try:
            # Open/create lock file
            self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as e:
            raise RuntimeError(f"Cannot open lock file {self._lock_path}: {e}")

        try:
            # Try to acquire exclusive non-blocking lock
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Lock is held by another process - read holder metadata
            holder_info = self._read_holder_metadata()
            os.close(self._fd)
            self._fd = None
            if holder_info:
                sys.stderr.write(
                    f"MPS heavy lock contention: held by {holder_info}\n"
                )
            else:
                sys.stderr.write(
                    f"MPS heavy lock contention: {self._lock_path} held by another process\n"
                )
            return False
        except OSError as e:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(f"Lock operation failed: {e}")

        # Lock acquired - write holder metadata
        self._write_holder_metadata()
        return True

    def _write_holder_metadata(self) -> None:
        """Write current holder metadata to lock file."""
        if self._fd is None:
            return
        metadata = (
            f"pid={self._pid}\n"
            f"milestone={self.milestone}\n"
            f"worktree={self.worktree}\n"
            f"command={self.command}\n"
            f"start_timestamp={self._start_ts:.6f}\n"
            f"hostname={self._hostname}\n"
        )
        try:
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, metadata.encode("utf-8"))
            os.fsync(self._fd)
        except OSError:
            # Non-fatal - metadata is best effort
            pass

    def _read_holder_metadata(self) -> Optional[str]:
        """Read holder metadata from lock file.

        Returns:
            Formatted string with holder info, or None if unreadable
        """
        try:
            with open(self._lock_path, "r") as f:
                content = f.read().strip()
            if not content:
                return None
            lines = content.split("\n")
            info = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k] = v
            return (
                f"pid={info.get('pid', '?')}, "
                f"milestone={info.get('milestone', '?')}, "
                f"worktree={info.get('worktree', '?')}, "
                f"command={info.get('command', '?')[:80]}, "
                f"start_ts={info.get('start_timestamp', '?')}, "
                f"hostname={info.get('hostname', '?')}"
            )
        except Exception:
            return None

    def release(self) -> None:
        """Release the lock explicitly."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "MPSHeavyLock":
        """Context manager entry - acquires lock or raises."""
        if not self.acquire():
            raise RuntimeError(
                f"Failed to acquire MPS heavy lock (held by another process). "
                f"Lock path: {self._lock_path}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - always releases lock."""
        self.release()

    @staticmethod
    def get_lock_path() -> Path:
        """Get the frozen lock path."""
        return _LOCK_PATH

    @staticmethod
    def check_lock_held() -> Optional[str]:
        """Check if lock is currently held and return holder info.

        Returns:
            Holder metadata string if held, None if free or unreadable
        """
        try:
            fd = os.open(_LOCK_PATH, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # We got it, so it wasn't held - release immediately
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return None
            except BlockingIOError:
                # Held by another process
                os.close(fd)
                # Read metadata
                with open(_LOCK_PATH, "r") as f:
                    content = f.read().strip()
                if not content:
                    return "held (no metadata)"
                return content.replace("\n", ", ")
            except OSError:
                os.close(fd)
                return None
        except FileNotFoundError:
            return None
        except Exception:
            return None


def acquire_mps_heavy_lock(
    milestone: str,
    worktree: str,
    command: str,
) -> MPSHeavyLock:
    """Convenience function to acquire lock or raise.

    Args:
        milestone: Milestone identifier
        worktree: Worktree path
        command: Command being executed

    Returns:
        Acquired MPSHeavyLock instance

    Raises:
        RuntimeError: If lock cannot be acquired
    """
    lock = MPSHeavyLock(milestone, worktree, command)
    if not lock.acquire():
        raise RuntimeError(
            f"MPS heavy lock unavailable - another heavy MPS job is running. "
            f"Lock path: {_LOCK_PATH}"
        )
    return lock


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="MPS Heavy Lock Test Utility")
    parser.add_argument("--milestone", default="TEST", help="Milestone identifier")
    parser.add_argument("--worktree", default=".", help="Worktree path")
    parser.add_argument("--command", default="test", help="Command string")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="Hold duration")
    parser.add_argument("--check-only", action="store_true", help="Just check lock status")

    args = parser.parse_args()

    if args.check_only:
        holder = MPSHeavyLock.check_lock_held()
        if holder:
            print(f"LOCK HELD: {holder}")
            sys.exit(1)
        else:
            print("LOCK FREE")
            sys.exit(0)

    lock = MPSHeavyLock(args.milestone, args.worktree, args.command)
    if lock.acquire():
        print(f"Lock acquired by PID {os.getpid()}")
        time.sleep(args.hold_seconds)
        lock.release()
        print("Lock released")
    else:
        print("Failed to acquire lock")
        sys.exit(1)