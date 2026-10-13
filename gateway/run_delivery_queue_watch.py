"""Wake the restart-safe cron delivery drain when a worker writes the queue (#117307).

A restart-safe worker runs its job outside the gateway, queues the final send in
``cron/deliveries.db`` and polls that row once a second; the gateway drained the
queue only on its housekeeping tick, so a turn that ended in 8 s reached the user
a minute later. Between ticks the housekeeping thread now stats the queue file
(and its WAL) and drains as soon as either changed. The tick's own drain stays
as the fallback for anything the stat cannot see.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

# How often the housekeeping thread looks at the queue file between its ticks.
DELIVERY_QUEUE_WATCH_SECONDS = 1.0
DRAIN_LABEL = "Cron durable delivery queue drain"


class DeliveryQueueWatch:
    """Stamp-compare every served profile's queue file; ``drain`` when a stamp moved.

    The watch never resets itself after draining: the drain's own status writes
    move the stamp once more and cost one empty pass, after which the stamp is
    stable, while a worker that enqueued during the drain still moves it again.
    """

    def __init__(self, homes: Callable[[], Iterable[Optional[Path]]], drain: Callable[[], None]) -> None:
        self._homes = homes
        self.drain = drain
        self._paths: tuple[str, ...] = ()
        self.refresh_homes()
        self._seen = self._stamp()

    def refresh_homes(self) -> None:
        """Re-resolve the served profiles and their queue files once per tick (a directory read under
        multiplex, a home lookup and a realpath per home), so each poll is only the two stats; a profile
        that appears mid-interval is drained by the tick itself."""
        from cron.delivery_queue import queue_path

        paths = []
        for home in self._homes():
            path = queue_path(home)
            paths += [str(path), str(path.with_name(path.name + "-wal"))]
        self._paths = tuple(paths)

    def _stamp(self) -> tuple:
        stamps = []
        for candidate in self._paths:
            try:
                info = os.stat(candidate)
            except OSError:
                continue
            stamps.append((candidate, info.st_mtime_ns, info.st_size))
        return tuple(stamps)

    def changed(self) -> bool:
        current = self._stamp()
        if current == self._seen:
            return False
        self._seen = current
        return True


def wait_for_next_tick(stop_event, interval: float, watch: Optional[DeliveryQueueWatch], chore) -> None:
    """Sleep one housekeeping interval, draining the worker queue the moment it changes."""
    if watch is None:
        stop_event.wait(timeout=interval)
        return
    watch.refresh_homes()
    deadline = time.monotonic() + interval
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        if stop_event.wait(timeout=min(remaining, DELIVERY_QUEUE_WATCH_SECONDS)):
            return
        if watch.changed():
            chore(DRAIN_LABEL, watch.drain)
        if time.monotonic() >= deadline:
            return
