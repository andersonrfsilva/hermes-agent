"""Messaging-gateway ownership tests for the hosted Group Chat worker."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver, hosted_rooms
from gateway.run import GatewayRunner
from tui_gateway.hosted_room_service import HostedRoomService
from tests.waiting import async_wait_until


class _RPC:
    def __init__(self) -> None:
        self.sessions = {}
        self.submits = []

    def resolve_exact(self, *, profile, title, source):
        del source
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        del source
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        del profile, source
        return {"session_id": session_id}

    def submit(self, **kwargs):
        self.submits.append(kwargs["profile"])
        kwargs["on_terminal"]({
            "status": "settled",
            "text": f"reply from {kwargs['profile']}",
        })
        return {"accepted": True}

    def history(self, **kwargs):
        del kwargs
        return []

    def info(self, **kwargs):
        del kwargs
        return {"active": False, "task_id": None}

    def interrupt(self, **kwargs):
        del kwargs
        raise AssertionError("gateway lifecycle must not interrupt room work")


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _service(db_path, *, profiles=("default",)):
    service = HostedRoomService(_server(), db_path=db_path)
    rpc = _RPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: profiles
    return service, rpc


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not settle before timeout")


@pytest.mark.asyncio
async def test_messaging_gateway_supervisor_starts_without_dashboard(monkeypatch):
    from tui_gateway import methods_groups

    state = {"running": False, "starts": 0}

    class Runtime:
        def status(self):
            return {"running": state["running"], "stopping": False}

    service = SimpleNamespace(runtime=Runtime())

    def get_service():
        return service if state["running"] else None

    def start_service():
        state["starts"] += 1
        state["running"] = True
        return service

    monkeypatch.setattr(methods_groups, "get_hosted_room_service", get_service)
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", start_service)

    runner = GatewayRunner.__new__(GatewayRunner)
    started = await runner._ensure_hosted_room_worker()
    assert started is service
    assert state == {"running": True, "starts": 1}

    # A dead child is restarted, while a healthy one is left alone.
    await runner._ensure_hosted_room_worker()
    assert state["starts"] == 1
    state["running"] = False
    await runner._ensure_hosted_room_worker()
    assert state["starts"] == 2


@pytest.mark.asyncio
async def test_dead_room_worker_is_restarted_by_gateway_task_supervision(monkeypatch):
    from tui_gateway import methods_groups

    starts = {"count": 0}

    def fail_start():
        starts["count"] += 1
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(methods_groups, "get_hosted_room_service", lambda: None)
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", fail_start)
    monkeypatch.setattr(hosted_rooms, "has_live_rooms", lambda _db: True)
    monkeypatch.setattr(GatewayRunner, "_MAX_SUPERVISED_RESTARTS", 1)
    monkeypatch.setattr(
        GatewayRunner,
        "_supervised_backoff",
        staticmethod(lambda _attempt: 0),
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    runner._spawn_supervised(
        lambda: runner._hosted_room_worker_watcher(interval=0),
        "hosted_room_worker",
    )

    for _ in range(200):
        if starts["count"] == 2 and not runner._background_tasks:
            break
        await asyncio.sleep(0.01)
    runner._running = False

    assert starts["count"] == 2
    assert runner._background_tasks == set()


@pytest.mark.asyncio
async def test_idle_gateway_wants_room_worker_only_while_a_live_room_exists(tmp_path, monkeypatch):
    db = tmp_path / "shared-state.db"
    monkeypatch.setattr(hosted_rooms, "default_db_path", lambda: db)
    members = [{"member_id": "default", "profile": "default", "handle": "hermes"}]

    # No store yet: no worker, and the probe must not create the file.
    assert not await GatewayRunner.__new__(GatewayRunner)._hosted_room_worker_wanted()
    assert not db.exists()

    # A live room makes the worker wanted.
    room = hosted_rooms.create_room(db, room_id="room-1", name="Release", members=members,
                                    authority_gateway_id="gw-1")
    runner = GatewayRunner.__new__(GatewayRunner)
    assert await runner._hosted_room_worker_wanted()
    # "Wanted" is never cached, so a failed start keeps being retried.
    assert await runner._hosted_room_worker_wanted()

    hosted_rooms.disband_room(db, room_id="room-1", expected_gateway_id="gw-1",
                              expected_epoch=room["authority_epoch"])
    idle = GatewayRunner.__new__(GatewayRunner)
    assert not await idle._hosted_room_worker_wanted()
    # A room created after the gateway settled on "no rooms" is picked up on the next probe.
    hosted_rooms.create_room(db, room_id="room-2", name="Ops", members=members,
                             authority_gateway_id="gw-1")
    assert await idle._hosted_room_worker_wanted()


@pytest.mark.asyncio
async def test_watcher_starts_room_worker_once_the_first_room_appears(tmp_path, monkeypatch):
    db = tmp_path / "shared-state.db"
    monkeypatch.setattr(hosted_rooms, "default_db_path", lambda: db)
    starts, probes = [], []
    monkeypatch.setattr(GatewayRunner, "_start_hosted_room_worker_sync", staticmethod(lambda: starts.append(1)))
    probe = GatewayRunner._hosted_room_store_signature
    monkeypatch.setattr(GatewayRunner, "_hosted_room_store_signature",
                        staticmethod(lambda: probes.append(1) or probe()))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    watcher = asyncio.create_task(runner._hosted_room_worker_watcher(interval=0.01))
    try:
        await async_wait_until(lambda: len(probes) >= 3, what="watcher probed the empty store")
        assert starts == []  # empty home: the worker (and tui_gateway.server) never loads
        hosted_rooms.create_room(db, room_id="room-1", name="Release", members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"}], authority_gateway_id="gw-1")
        await async_wait_until(lambda: starts, what="worker started for the new room")
    finally:
        runner._running = False
        await watcher


def test_live_room_probe_reads_legacy_store_only_until_its_import_is_recorded(tmp_path):
    members = [{"member_id": "default", "profile": "default", "handle": "hermes"}]

    # Upgraded install: rooms still only in the pre-isolation state.db, shared store not opened yet.
    pending = tmp_path / "pending"
    pending.mkdir()
    hosted_rooms.create_room(pending / "state.db", room_id="old", name="Old", members=members,
                             authority_gateway_id="gw-1")
    assert hosted_rooms.has_live_rooms(pending / "shared-state.db")

    # Import already recorded (the copy leaves the source intact): the legacy rows are history.
    settled = tmp_path / "settled"
    settled.mkdir()
    shared = settled / "shared-state.db"
    room = hosted_rooms.create_room(shared, room_id="room-1", name="New", members=members,
                                    authority_gateway_id="gw-1")
    hosted_rooms.disband_room(shared, room_id="room-1", expected_gateway_id="gw-1",
                              expected_epoch=room["authority_epoch"])
    hosted_rooms.create_room(settled / "state.db", room_id="old", name="Old", members=members,
                             authority_gateway_id="gw-1")
    assert not hosted_rooms.has_live_rooms(shared)


def test_gateway_restart_resumes_queued_room_for_multiplexed_profile(tmp_path):
    db = tmp_path / "state.db"
    first, _ = _service(db, profiles=("default", "ops"))
    first.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    first.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    assert (
        len(hosted_room_driver.list_tasks(db, room_id="room-1", status="queued")) == 1
    )

    resumed, rpc = _service(db, profiles=("default", "ops"))
    resumed.start()
    try:
        _wait_for(
            lambda: any(
                event["kind"] == "message.member"
                for event in hosted_rooms.read_events(
                    db, room_id="room-1", since_seq=0
                )["events"]
            )
        )
    finally:
        assert resumed.stop(timeout=5.0)

    assert rpc.submits == ["ops"]
    assert hosted_room_driver.list_tasks(db, room_id="room-1", status="settled")


def test_dashboard_and_gateway_workers_share_one_fenced_execution_owner(tmp_path):
    db = tmp_path / "state.db"
    gateway, gateway_rpc = _service(db, profiles=("default", "ops"))
    dashboard, dashboard_rpc = _service(db, profiles=("default", "ops"))
    gateway.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    gateway.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )

    gateway.start()
    dashboard.start()
    try:
        _wait_for(
            lambda: any(
                event["kind"] == "message.member"
                for event in hosted_rooms.read_events(
                    db, room_id="room-1", since_seq=0
                )["events"]
            )
        )
        time.sleep(0.05)
    finally:
        assert gateway.stop(timeout=5.0)
        assert dashboard.stop(timeout=5.0)

    assert len(gateway_rpc.submits) + len(dashboard_rpc.submits) == 1
    events = hosted_rooms.read_events(db, room_id="room-1", since_seq=0)["events"]
    assert sum(event["kind"] == "message.member" for event in events) == 1
