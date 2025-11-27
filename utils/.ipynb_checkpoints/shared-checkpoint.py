"""utils/shared.py - Shared utilities for tensor-based Go implementation.

This module contains common utilities used across the Go engine, agents, and
simulation. All utilities are self-contained and can be imported by any module.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from functools import wraps
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

import torch
from torch import Tensor

# Avoid circular imports
if TYPE_CHECKING:
    from engine.tensor_native import TensorBoard  # noqa: F401


# ========================= DEVICE UTILITIES =========================


def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU.

    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")









# ========================= TIMING UTILITIES =========================


def _sync_device(device: Optional[torch.device]) -> None:
    """Synchronize device if needed (CUDA / MPS)."""
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class TimingContext:
    """Context manager for timing operations with device synchronization.

    Used internally by @timed_method, which also handles nested timers and
    exclusive / inclusive accounting.
    """

    def __init__(
        self,
        timings: Dict[str, List[float]],
        exclusive_timings: Dict[str, List[float]],
        stack: List[Dict],
        name: str,
        device: torch.device,
        enable_timing: bool = True,
    ):
        self.timings = timings
        self.exclusive_timings = exclusive_timings
        self.stack = stack
        self.name = name
        self.device = device
        self.enable_timing = enable_timing
        self.start_time: Optional[float] = None

    def __enter__(self):
        if not self.enable_timing:
            return self

        _sync_device(self.device)
        now = time.perf_counter()

        # Push a frame onto the stack to track nested child time
        frame = {
            "name": self.name,
            "start": now,
            "child_time": 0.0,
        }
        self.stack.append(frame)
        self.start_time = now
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable_timing or self.start_time is None:
            return

        _sync_device(self.device)
        end = time.perf_counter()
        elapsed = end - self.start_time

        frame = self.stack.pop()
        assert frame["name"] == self.name, "Timing stack mismatch"

        child_time = frame["child_time"]
        exclusive = max(0.0, elapsed - child_time)

        # Inclusive time: full span of the call
        self.timings[self.name].append(elapsed)
        # Exclusive time: minus all timed children
        self.exclusive_timings[self.name].append(exclusive)

        # Propagate elapsed time up to parent as "child_time"
        if self.stack:
            self.stack[-1]["child_time"] += elapsed


def timed_method(method: Callable) -> Callable:
    """Decorator to time methods with nested-aware exclusive/inclusive accounting."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # If timing is explicitly disabled, just run.
        if hasattr(self, "enable_timing") and not getattr(self, "enable_timing", True):
            return method(self, *args, **kwargs)

        # Ensure timing structures exist on the instance
        if not hasattr(self, "timings"):
            self.timings = defaultdict(list)
        if not hasattr(self, "exclusive_timings"):
            self.exclusive_timings = defaultdict(list)
        if not hasattr(self, "call_counts"):
            self.call_counts = defaultdict(int)
        if not hasattr(self, "_timing_stack"):
            self._timing_stack: List[Dict] = []

        # Increment call count
        self.call_counts[method.__name__] += 1

        device = getattr(self, "device", None)
        ctx = TimingContext(
            timings=self.timings,
            exclusive_timings=self.exclusive_timings,
            stack=self._timing_stack,
            name=method.__name__,
            device=device,
            enable_timing=getattr(self, "enable_timing", True),
        )
        with ctx:
            return method(self, *args, **kwargs)

    return wrapper


def print_timing_report(
    obj,
    top_n: int = 30,
    total_wall_time: Optional[float] = None,
) -> None:
    """Pretty-print timing report for any object with timing attributes.

    Expects the object to have:
      - obj.timings: Dict[str, List[float]] (inclusive times)
      - obj.exclusive_timings: Dict[str, List[float]] (exclusive/self times)
      - obj.call_counts: Dict[str, int]

    If total_wall_time (seconds) is provided, prints an extra "%Wall" column.
    """
    timings: Dict[str, List[float]] = getattr(obj, "timings", {})
    exclusive_timings: Dict[str, List[float]] = getattr(obj, "exclusive_timings", {})
    call_counts: Dict[str, int] = getattr(obj, "call_counts", {})

    if not timings:
        print("No timing data collected.")
        return

    # Aggregate stats per function
    stats = []
    total_exclusive = 0.0
    for name, samples in timings.items():
        total_inclusive = sum(samples)
        calls = call_counts.get(name, len(samples))
        excl_samples = exclusive_timings.get(name, [])
        total_self = sum(excl_samples) if excl_samples else total_inclusive
        avg_self = total_self / calls if calls > 0 else 0.0

        stats.append(
            {
                "name": name,
                "total_inclusive": total_inclusive,
                "total_self": total_self,
                "avg_self": avg_self,
                "calls": calls,
            }
        )
        total_exclusive += total_self

    # Sort by inclusive time (heaviest overall)
    stats.sort(key=lambda s: s["total_inclusive"], reverse=True)

    print("\n" + "=" * 80)
    print("TIMING REPORT")
    print("=" * 80 + "\n")

    print(f"Top {top_n} Time-Consuming Functions:")
    header = (
        f"{'Function':<40} {'Total(ms)':>10} {'Self(ms)':>10} "
        f"{'AvgSelf(ms)':>12} {'Count':>8} {'%Self':>8}"
        + ("" if total_wall_time is None else " {:>10}".format("%Wall"))
    )
    print(header)
    print("-" * len(header))

    # Avoid division by zero
    total_exclusive_ms = total_exclusive * 1000.0 if total_exclusive > 0 else 1.0
    wall_ms = total_wall_time * 1000.0 if total_wall_time is not None else None

    for s in stats[:top_n]:
        total_ms = s["total_inclusive"] * 1000.0
        self_ms = s["total_self"] * 1000.0
        avg_self_ms = s["avg_self"] * 1000.0
        pct_self = 100.0 * self_ms / total_exclusive_ms

        row = (
            f"{s['name']:<40} "
            f"{total_ms:10.2f} {self_ms:10.2f} {avg_self_ms:12.4f} "
            f"{s['calls']:8d} {pct_self:8.1f}"
        )
        if wall_ms is not None:
            pct_wall = 100.0 * total_ms / wall_ms
            row += f" {pct_wall:10.1f}"
        print(row)

    print()

    if total_wall_time is not None:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total wall time: {total_wall_time:.3f} s")
        print(f"Total exclusive timed time: {total_exclusive:.3f} s")
        print(
            "Note: exclusive times sum to total timed work (no double counting); "
            "wall time can be smaller due to untimed code and I/O."
        )
        print()


# ========================= PRINTING / DEBUG UTILITIES =========================


def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)







def print_performance_metrics(
    elapsed: float,
    moves_made: int,
    num_games: int,
) -> None:
    """Print performance metrics for a simulation run."""
    print_section_header("PERFORMANCE METRICS")
    print(f"Total simulation time: {elapsed:.2f} seconds")
    print(f"Moves per second: {moves_made / elapsed:.1f}")
    print(f"Games per second: {num_games / elapsed:.1f}")
    print(f"Time per move: {elapsed / moves_made * 1000:.2f} ms")
    print(f"Time per game: {elapsed / num_games:.3f} seconds")


def print_game_summary(stats) -> None:
    """Print game statistics summary."""
    print(
        f"\nFinished {stats.total_games} games in {stats.duration_seconds:.2f}s "
        f"({stats.seconds_per_move:.4f}s/ply)"
    )
    print(f"Black wins: {stats.black_wins:4d} ({stats.black_win_rate:6.1%})")
    print(f"White wins: {stats.white_wins:4d} ({stats.white_win_rate:6.1%})")
    print(f"Draws     : {stats.draws:4d} ({stats.draw_rate:6.1%})")



