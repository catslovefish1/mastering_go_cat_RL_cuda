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


# ========================= COORDINATE UTILITIES =========================


def flat_to_2d(flat_indices: Tensor, width: int) -> Tuple[Tensor, Tensor]:
    """Convert flat indices to 2D coordinates.

    Args:
        flat_indices: Shape (N,) flat position indices.
        width: Board width for modulo operation.

    Returns:
        Tuple of (rows, cols) tensors.
    """
    rows = flat_indices // width
    cols = flat_indices % width
    return rows, cols


def coords_to_flat(rows: Tensor, cols: Tensor, width: int) -> Tensor:
    """Convert 2D coordinates to flat indices.

    Args:
        rows: Row coordinates.
        cols: Column coordinates.
        width: Board width.

    Returns:
        Flat indices tensor.
    """
    return rows * width + cols


# ========================= POSITION UTILITIES =========================


def create_pass_positions(batch_size: int, device: torch.device) -> Tensor:
    """Create tensor of pass moves for given batch size.

    Pass moves are represented as [-1, -1] in the position tensor.

    Args:
        batch_size: Number of positions to create.
        device: Target device for tensor creation.

    Returns:
        Tensor of shape (batch_size, 2) filled with -1.
    """
    return torch.full((batch_size, 2), -1, dtype=torch.int16, device=device)


# ========================= PROBABILITY UTILITIES =========================


def compute_uniform_probabilities(mask: Tensor) -> Tensor:
    """Compute uniform probability distribution over True values in mask.

    Args:
        mask: Shape (N, M) boolean mask.

    Returns:
        Shape (N, M) probability distribution (sums to 1 along dim=1).
    """
    probs = mask.float()
    row_sums = probs.sum(dim=1, keepdim=True)
    # Avoid division by zero: rows with no True entries stay all zeros
    safe_sums = row_sums.clamp(min=1.0)
    return probs / safe_sums


def sample_from_mask(mask: Tensor, num_samples: int = 1) -> Tensor:
    """Sample indices from a boolean mask with uniform probability.

    Args:
        mask: Shape (N, M) boolean mask.
        num_samples: Number of samples per row.

    Returns:
        Shape (N, num_samples) or (N,) if num_samples=1.
    """
    probabilities = compute_uniform_probabilities(mask)
    sampled = torch.multinomial(probabilities, num_samples=num_samples)
    if num_samples == 1:
        sampled = sampled.squeeze(1)
    return sampled


# ========================= BATCH UTILITIES =========================
# (placeholder for future shared batch helpers)


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


def print_union_find_grid(board, batch_idx: int = 0, column: int = 0) -> None:
    """Print union-find data in grid format.

    Args:
        board: Object with `board_size` and `flatten_union_find`.
        batch_idx: Batch index to print.
        column: 0=Colour, 1=Parent, 2=Liberty.
    """
    column_names = ["Colour", "Parent", "Liberty"]
    print(f"\n{column_names[column]} values for batch {batch_idx}:")
    print("-" * (board.board_size * 4 + 1))

    uf_data = board.flatten_union_find[batch_idx, :, column].view(
        board.board_size, board.board_size
    )

    for row in range(board.board_size):
        row_str = "|"
        for col in range(board.board_size):
            value = uf_data[row, col].item()
            row_str += f"{value:3}|"
        print(row_str)
    print("-" * (board.board_size * 4 + 1))


def print_all_union_find_columns(
    board,
    batch_idx: int = 0,
    board_size_limit: int = 9,
) -> None:
    """Print all union-find columns with appropriate formatting.

    Always prints colour; parent/liberty only for small boards.
    """
    print("\nCOLOUR (-1=empty, 0=black, 1=white):")
    print_union_find_grid(board, batch_idx, column=0)

    if board.board_size <= board_size_limit:
        print("\nPARENT INDICES:")
        print_union_find_grid(board, batch_idx, column=1)

        print("\nLIBERTY COUNTS:")
        print_union_find_grid(board, batch_idx, column=2)


def print_move_info(move: Tensor, player: int) -> None:
    """Print information about a move."""
    player_name = "BLACK" if player == 0 else "WHITE"
    print(f"\nCurrent player: {player_name} ({player})")
    print(f"Move to be played: {move.tolist()}")

    if move[0] >= 0:  # Not a pass
        print(f"  Position: row={move[0].item()}, col={move[1].item()}")
    else:
        print("  PASS MOVE")


def print_game_state(
    board,
    batch_idx: int = 0,
    ply: int = 0,
    header: str = "",
    move: Optional[Tensor] = None,
) -> None:
    """Print complete game state for debugging."""
    print_section_header(f"{header} - Ply {ply}")

    if move is not None:
        print_move_info(move, board.current_player[batch_idx].item())

    print_all_union_find_columns(board, batch_idx)

    print(f"\nPass count: {board.pass_count[batch_idx].item()}")
    if hasattr(board, "ko_points") and board.ko_points[batch_idx, 0] >= 0:
        ko_row = board.ko_points[batch_idx, 0].item()
        ko_col = board.ko_points[batch_idx, 1].item()
        print(f"Ko point: ({ko_row}, {ko_col})")


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


# ========================= GAME HISTORY DUMP =========================


def save_game_histories_to_json(
    boards,
    num_games_to_save: int = 5,
    output_dir: str = "game_histories",
) -> None:
    """Dump up to *num_games_to_save* finished games from a TensorBoard batch.

    Uses the refactored data structure:

        boards.board         – (B, H, W) int8   (-1 empty, 0 black, 1 white)
        boards.board_history – (B, T, N2) int8  snapshots taken *before* each move
        boards.hash_history  – (B, T)   int32   Zobrist hash of the same snapshots
        boards.current_hash  – (B,)     int32   hash of the *current* board

    Hash alignment:
    - For move m (1-based in the JSON), we emit:
        pre_hash  = hash of the board before move m  (hash_history[g, m-1])
        post_hash = hash of the board after  move m  (hash_history[g, m]
                                                      if exists, else current_hash[g])
    """
    os.makedirs(output_dir, exist_ok=True)

    B = boards.batch_size
    max_moves = boards.board_history.shape[1]
    N = boards.board_size
    board_area = N * N
    dev = boards.device

    has_hash_hist = hasattr(boards, "hash_history") and boards.hash_history is not None
    has_curr_hash = hasattr(boards, "current_hash") and boards.current_hash is not None

    EMPTY = torch.full((board_area,), -1, dtype=torch.int8, device=dev)

    for g in range(min(num_games_to_save, B)):
        total_moves = int(boards.move_count[g].item())
        recorded = min(total_moves, max_moves)

        prev = EMPTY
        moves = []

        for m in range(recorded):
            # Pre-move board snapshot at ply m (0-based)
            curr = boards.board_history[g, m]  # (N2,) int8

            pre_hash = int(boards.hash_history[g, m].item()) if has_hash_hist else None
            if has_hash_hist:
                if m + 1 < max_moves and (m + 1) < recorded:
                    post_hash = int(boards.hash_history[g, m + 1].item())
                else:
                    post_hash = (
                        int(boards.current_hash[g].item()) if has_curr_hash else None
                    )
            else:
                post_hash = None

            diff = (curr != prev).nonzero(as_tuple=True)[0]

            move_descr: Dict[str, object] = {
                "move_number": m + 1,
                "board_state": curr.detach().cpu().tolist(),
                "changes": [],
                "pre_hash": pre_hash,
                "post_hash": post_hash,
            }

            if diff.numel() == 0:
                # Pure pass
                move_descr["changes"].append({"type": "pass"})
            else:
                for pos in diff.tolist():
                    row, col = divmod(int(pos), N)
                    old = int(prev[pos].item())
                    new = int(curr[pos].item())

                    if old == -1 and new != -1:
                        move_descr["changes"].append(
                            {
                                "type": "place",
                                "position": [row, col],
                                "color": "black" if new == 0 else "white",
                            }
                        )
                    elif old != -1 and new == -1:
                        move_descr["changes"].append(
                            {
                                "type": "capture",
                                "position": [row, col],
                                "color": "black" if old == 0 else "white",
                            }
                        )
                    else:
                        move_descr["changes"].append(
                            {
                                "type": "flip",
                                "position": [row, col],
                                "from": "empty"
                                if old == -1
                                else ("black" if old == 0 else "white"),
                                "to": "empty"
                                if new == -1
                                else ("black" if new == 0 else "white"),
                            }
                        )

            moves.append(move_descr)
            prev = curr.clone()

        final_board = boards.board[g]  # (H, W) int8
        black_stones = int((final_board == 0).sum().item())
        white_stones = int((final_board == 1).sum().item())

        final_hash = int(boards.current_hash[g].item()) if has_curr_hash else None

        game_json = {
            "game_id": g,
            "board_size": N,
            "total_moves": total_moves,
            "moves_recorded": recorded,
            "truncated": total_moves > max_moves,
            "final_score": {"black": black_stones, "white": white_stones},
            "final_hash": final_hash,
            "moves": moves,
        }

        path = os.path.join(output_dir, f"game_{g:03d}.json")
        with open(path, "w") as f:
            json.dump(game_json, f, indent=2)

