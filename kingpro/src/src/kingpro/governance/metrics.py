"""Small dependency-free operational metrics registry for one API process."""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict, deque


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[index], 3)


class MetricsRegistry:
    def __init__(self, window: int = 2000) -> None:
        self._counts: Counter[tuple[str, str, int]] = Counter()
        self._latencies: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.Lock()

    def observe(self, method: str, path: str, status: int, elapsed_ms: float) -> None:
        route = path.split("?", 1)[0]
        with self._lock:
            self._counts[(method, route, int(status))] += 1
            self._latencies[(method, route)].append(float(elapsed_ms))

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            latencies = {key: list(values) for key, values in self._latencies.items()}
        total = sum(counts.values())
        errors = sum(value for (_, _, status), value in counts.items() if status >= 500)
        routes = []
        for (method, route), values in sorted(latencies.items()):
            route_counts = {
                str(status): count
                for (m, r, status), count in counts.items()
                if m == method and r == route
            }
            routes.append(
                {
                    "method": method,
                    "route": route,
                    "requests": sum(route_counts.values()),
                    "status_counts": route_counts,
                    "latency_ms": {
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                        "max": round(max(values), 3) if values else 0.0,
                    },
                }
            )
        error_rate = errors / total if total else 0.0
        return {
            "requests": total,
            "server_errors": errors,
            "server_error_rate": round(error_rate, 6),
            "routes": routes,
            "alerts": ["server_error_rate_above_5pct"] if total >= 20 and error_rate > 0.05 else [],
            "claim_limit": "single-process rolling operational telemetry",
        }


METRICS = MetricsRegistry()

