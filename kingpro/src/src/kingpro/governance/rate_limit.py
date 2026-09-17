"""Bounded in-memory token bucket for a single product process."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, policy: str, *, capacity: int, per_seconds: float) -> tuple[bool, float]:
        now = time.monotonic()
        refill = capacity / per_seconds
        bucket_key = (str(key), str(policy))
        with self._lock:
            bucket = self._buckets.get(bucket_key, _Bucket(float(capacity), now))
            bucket.tokens = min(float(capacity), bucket.tokens + (now - bucket.updated) * refill)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                self._buckets[bucket_key] = bucket
                return True, 0.0
            retry = (1.0 - bucket.tokens) / refill
            self._buckets[bucket_key] = bucket
            return False, retry

