"""Idle gates for the gateway's per-profile pollers (heartbeat restore, handoff watcher, loop wakeup).

Entering ``_profile_runtime_scope`` costs a config.yaml load, a ``.env`` parse, secret hydration and a
terminal-policy build; on a multiplex gateway the pollers paid that per profile per tick with nothing
to do. Each gate answers "does this profile's store hold work?" from the goals-cached SessionDB with
ONLY the HERMES_HOME contextvar installed. Every gate fails OPEN: an unavailable store, a failing
read or a corrupt row is "cannot prove emptiness", never "idle".
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.run")


def _profile_session_db_probe(profile_home: Path) -> Optional[Any]:
    """The goals-cached SessionDB for *profile_home*; None when unavailable."""
    from hermes_cli.goals import _get_session_db
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    try:
        return _get_session_db()
    except Exception:
        logger.debug("session-db probe failed for %s", profile_home, exc_info=True)
        return None
    finally:
        reset_hermes_home_override(token)


def _gate(profile_home: Path, store_has_work: Callable[[Any], bool]) -> bool:
    db = _profile_session_db_probe(profile_home)
    if db is None:
        return True
    try:
        return store_has_work(db)
    except Exception:
        logger.debug("idle probe failed for %s; keeping the full sweep", profile_home, exc_info=True)
        return True


def profile_has_active_heartbeat(profile_home: Path) -> bool:
    from hermes_cli.heartbeat import store_has_active_heartbeat

    return _gate(profile_home, store_has_active_heartbeat)


def profile_has_active_loop(profile_home: Path) -> bool:
    from hermes_cli.loops import store_has_active_loop

    return _gate(profile_home, store_has_active_loop)


def profile_has_pending_handoff(profile_home: Path) -> bool:
    return _gate(profile_home, lambda db: db.has_pending_handoffs())


async def off_loop_gate(runner: object, probe: Callable[[], bool]) -> bool:
    """Run a sync gate through the runner's executor hop. Runners without one (bare test stand-ins
    for the handoff watcher) keep the historical always-enter behaviour."""
    offload = getattr(runner, "_run_in_executor_with_context", None)
    if not callable(offload):
        return True
    return bool(await offload(probe))


def handoff_store_path(profile_home: Optional[Path]) -> Path:
    """The ``state.db`` a handoff poll of *profile_home* reads (None: the process's own store)."""
    if profile_home is None:
        from hermes_state import _default_db_path

        return _default_db_path()
    return Path(profile_home) / "state.db"


def _store_signature(db_path: Path) -> Optional[tuple]:
    """``(mtime_ns, size)`` of a SQLite store and its WAL; None when the main file is missing."""
    signature = []
    for candidate in (db_path, db_path.with_name(db_path.name + "-wal")):
        try:
            st = os.stat(candidate)
        except OSError:
            if candidate is db_path:
                return None
            signature.append(None)
            continue
        signature.append((st.st_mtime_ns, st.st_size))
    return tuple(signature)


class StoreChangeGate:
    """Skip a poll while its SQLite store is untouched since a read PROVED it held nothing pending.

    A new row can only appear through a write, which moves the main file's or the WAL's
    ``(mtime_ns, size)``. ``check`` takes the signature BEFORE the poll and ``mark_empty`` records it
    only after a raising probe answered "nothing pending", so a write racing the poll, a failed read
    and a read that swallowed its error all keep polling. A store that cannot be stat'ed is always
    polled, and a proven-empty store is still re-read every ``max_skip_s``."""

    def __init__(self, max_skip_s: float = 60.0) -> None:
        self._max_skip_s = max_skip_s
        self._empty: dict[str, tuple[tuple, float]] = {}

    def check(self, db_path: Path) -> tuple[bool, Optional[tuple]]:
        """``(skip, signature)``: skip when the store is unchanged since it was proven empty."""
        signature = _store_signature(db_path)
        seen = self._empty.get(str(db_path))
        fresh = seen is not None and time.monotonic() - seen[1] < self._max_skip_s
        return signature is not None and fresh and seen[0] == signature, signature

    def mark_empty(self, db_path: Path, signature: Optional[tuple]) -> None:
        if signature is not None:
            self._empty[str(db_path)] = (signature, time.monotonic())


async def store_proven_empty(session_db: Any) -> bool:
    """True only when the scoped store answers "no pending handoff" without error; a failing probe
    or a stand-in without it is "unknown", never empty (``list_pending_handoffs`` swallows errors)."""
    probe = getattr(session_db, "has_pending_handoffs", None)
    if probe is None:
        return False
    try:
        return not await probe()
    except Exception:
        logger.debug("pending-handoff probe failed; the store stays unproven", exc_info=True)
        return False
