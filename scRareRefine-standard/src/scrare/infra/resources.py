from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceMonitor:
    sample_interval_seconds: float = 1.0
    _start_time: float = field(init=False, default=0.0)
    _end_time: float = field(init=False, default=0.0)
    _peak_rss_bytes: int = field(init=False, default=0)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)

    def __enter__(self) -> "ResourceMonitor":
        self._start_time = time.perf_counter()
        self._end_time = 0.0
        self._peak_rss_bytes = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sample_once()
        self._end_time = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.sample_interval_seconds * 2))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.sample_interval_seconds)

    def _sample_once(self) -> None:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(rss))

    def summary(self) -> dict[str, float]:
        end = self._end_time or time.perf_counter()
        return {
            "wall_time_seconds": float(end - self._start_time),
            "peak_rss_mb": float(self._peak_rss_bytes / (1024 * 1024)),
        }


__all__ = ["ResourceMonitor"]
