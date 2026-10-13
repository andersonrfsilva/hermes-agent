"""The handoff watcher does not re-read an untouched state.db, yet still picks up a new handoff.

An idle gateway polled ``list_pending_handoffs`` every 2 s (a thread hop plus a SQLite query) with
nothing to find. A pending handoff can only appear through a write to the store, so the watcher
now skips the read while the store and its WAL are unchanged.
"""

import asyncio
import types

import pytest

import hermes_state
from gateway import run
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_idle_store_is_not_reread_until_a_handoff_is_written(tmp_path, monkeypatch):
    from hermes_state import AsyncSessionDB, SessionDB

    store = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", store)
    db = SessionDB(store)
    db.create_session("s1", "cli")
    reads = []
    list_pending = db.list_pending_handoffs
    db.list_pending_handoffs = lambda: reads.append(1) or list_pending()

    real_sleep = asyncio.sleep

    async def _yield(_seconds):  # the 5 s warm-up and the 2 s interval, without wall time
        await real_sleep(0.005)

    monkeypatch.setattr(run.asyncio, "sleep", _yield)
    processed = []
    fake = types.SimpleNamespace(_session_db=AsyncSessionDB(db), _running=True)

    async def _process_handoff(row):
        processed.append(row["id"])
        fake._running = False

    fake._process_handoff = _process_handoff
    watcher = asyncio.create_task(run.GatewayRunner._handoff_watcher(fake, interval=0.0))
    try:
        await async_wait_until(lambda: reads, what="first handoff poll")
        idle_reads = len(reads)
        for _ in range(20):
            await real_sleep(0.005)
        assert len(reads) == idle_reads, "an untouched store was re-read"

        db.request_handoff("s1", "telegram")
        await async_wait_until(lambda: processed, what="handoff picked up after the write")
        assert processed == ["s1"]
    finally:
        fake._running = False
        await asyncio.wait_for(watcher, timeout=5)


@pytest.mark.asyncio
async def test_a_read_that_cannot_prove_emptiness_keeps_polling(tmp_path, monkeypatch):
    """``list_pending_handoffs`` turns a read error into ``[]``; that must not let the gate skip."""
    from hermes_state import AsyncSessionDB, SessionDB

    store = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", store)
    db = SessionDB(store)
    reads = []
    db.list_pending_handoffs = lambda: reads.append(1) or []  # a swallowed read error looks like this

    def _probe_fails():
        raise RuntimeError("database is locked")

    db.has_pending_handoffs = _probe_fails
    real_sleep = asyncio.sleep

    async def _yield(_seconds):
        await real_sleep(0.005)

    monkeypatch.setattr(run.asyncio, "sleep", _yield)
    fake = types.SimpleNamespace(_session_db=AsyncSessionDB(db), _running=True)

    async def _process_handoff(row):
        raise AssertionError("nothing is pending")

    fake._process_handoff = _process_handoff
    watcher = asyncio.create_task(run.GatewayRunner._handoff_watcher(fake, interval=0.0))
    try:
        await async_wait_until(lambda: len(reads) >= 3, what="unproven store re-read on every tick")
    finally:
        fake._running = False
        await asyncio.wait_for(watcher, timeout=5)
