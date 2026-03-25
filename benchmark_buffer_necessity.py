#!/usr/bin/env python3
"""
Benchmark pooled workspace necessity using existing duel workloads.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import duel_random_random_stable as duel_stable
import duel_random_vs_random as duel_vs_random


ROW_RE = re.compile(
    r"^([A-Za-z0-9_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s+([0-9.]+)$"
)
FLOAT_LINE_RE = re.compile(r"^([^:]+):\s+([0-9.]+)")
CUDA_LINE_RE = re.compile(
    r"^\[CUDA\]\[FINAL\]\s+(peak_alloc|peak_reserved|current|reserved)=([0-9.]+)\s+MB"
)


@dataclass(frozen=True)
class Mode:
    name: str
    env: dict[str, str]


@dataclass(frozen=True)
class Workload:
    script_name: str
    runner: Callable
    board_size: int
    num_games: int
    max_plies: int


def _set_mode_env(mode: Mode) -> dict[str, str | None]:
    keys = [
        "GO_ENGINE_ALLOC_PER_CALL_CHECKER",
        "GO_ENGINE_NO_CACHE_LATEST",
    ]
    old = {k: os.getenv(k) for k in keys}
    for k in keys:
        os.environ[k] = mode.env.get(k, "0")
    return old


def _restore_env(old_env: dict[str, str | None]) -> None:
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _parse_timing_reports(output: str) -> list[dict[str, dict[str, float]]]:
    reports: list[dict[str, dict[str, float]]] = []
    current: dict[str, dict[str, float]] | None = None

    for raw in output.splitlines():
        line = raw.strip()

        if line.startswith("Top ") and "Time-Consuming Functions" in line:
            current = {}
            reports.append(current)
            continue

        if current is None:
            continue

        if not line:
            if current:
                current = None
            continue

        m = ROW_RE.match(line)
        if not m:
            continue

        name = m.group(1)
        current[name] = {
            "total_ms": float(m.group(2)),
            "self_ms": float(m.group(3)),
            "avg_self_ms": float(m.group(4)),
            "calls": float(m.group(5)),
            "pct_self": float(m.group(6)),
        }

    return reports


def _parse_metrics(output: str) -> dict:
    parsed: dict = {
        "performance": {},
        "cuda_mb": {},
        "timing_reports": [],
    }

    reports = _parse_timing_reports(output)
    parsed["timing_reports"] = reports

    for raw in output.splitlines():
        line = raw.strip()
        m = FLOAT_LINE_RE.match(line)
        if m and m.group(1) in {
            "Total simulation time",
            "Moves per second",
            "Games per second",
            "Time per move",
            "Time per game",
        }:
            parsed["performance"][m.group(1)] = float(m.group(2))
            continue

        cm = CUDA_LINE_RE.match(line)
        if cm:
            parsed["cuda_mb"][cm.group(1)] = float(cm.group(2))

    return parsed


def _run_workload(
    workload: Workload,
    mode: Mode,
    device_kind: str,
    seed: int,
) -> dict:
    module = workload.runner.__module__
    if module == "duel_random_random_stable":
        module_obj = duel_stable
    elif module == "duel_random_vs_random":
        module_obj = duel_vs_random
    else:
        raise RuntimeError(f"unsupported module: {module}")

    if device_kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    old_env = _set_mode_env(mode)
    old_select_device = module_obj.select_device

    try:
        module_obj.select_device = lambda: torch.device(device_kind)

        # Keep workloads deterministic across modes for fair A/B comparison.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device_kind == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        history_dir = "debug_games_bench_tmp"
        captured = io.StringIO()

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(captured):
            workload.runner(
                num_games=workload.num_games,
                board_size=workload.board_size,
                max_plies=workload.max_plies,
                komi=0,
                log_interval=0,
                enable_timing=True,
                num_games_to_save=0,
                history_dir=history_dir,
            )
        if device_kind == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        output = captured.getvalue()
        parsed = _parse_metrics(output)

        actions_total = workload.num_games * workload.max_plies
        out = {
            "mode": mode.name,
            "device": device_kind,
            "script": workload.script_name,
            "board_size": workload.board_size,
            "num_games": workload.num_games,
            "max_plies": workload.max_plies,
            "actions_total": actions_total,
            "external_elapsed_s": t1 - t0,
            "external_actions_per_s": actions_total / max(t1 - t0, 1e-9),
            "parsed": parsed,
        }

        if len(parsed["timing_reports"]) >= 1:
            out["engine_timing"] = parsed["timing_reports"][0]
        else:
            out["engine_timing"] = {}

        if len(parsed["timing_reports"]) >= 2:
            out["checker_timing"] = parsed["timing_reports"][1]
        else:
            out["checker_timing"] = {}

        return out
    finally:
        module_obj.select_device = old_select_device
        _restore_env(old_env)


def _extract_self_ms(timing: dict, name: str) -> float:
    if name not in timing:
        return 0.0
    return float(timing[name].get("self_ms", 0.0))


def _summarize(results: list[dict]) -> dict:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in results:
        key = (
            row["device"],
            row["script"],
            row["board_size"],
            row["num_games"],
            row["max_plies"],
        )
        grouped.setdefault(key, {})[row["mode"]] = row

    summary_rows = []
    for key, modes in grouped.items():
        if "baseline" not in modes:
            continue
        base = modes["baseline"]
        base_thr = base["external_actions_per_s"]
        base_hash = _extract_self_ms(base["engine_timing"], "_build_candidate_hashes")
        base_compute = _extract_self_ms(base["engine_timing"], "_compute_legal_and_candidates")
        base_checker = _extract_self_ms(base["checker_timing"], "compute_batch_legal_and_info")

        for mode_name, row in modes.items():
            if mode_name == "baseline":
                continue
            thr = row["external_actions_per_s"]
            hash_self = _extract_self_ms(row["engine_timing"], "_build_candidate_hashes")
            compute_self = _extract_self_ms(row["engine_timing"], "_compute_legal_and_candidates")
            checker_self = _extract_self_ms(row["checker_timing"], "compute_batch_legal_and_info")

            summary_rows.append(
                {
                    "device": key[0],
                    "script": key[1],
                    "board_size": key[2],
                    "num_games": key[3],
                    "max_plies": key[4],
                    "mode": mode_name,
                    "throughput_delta_pct": 100.0 * (thr - base_thr) / max(base_thr, 1e-9),
                    "hash_self_delta_pct": 100.0 * (hash_self - base_hash) / max(base_hash, 1e-9),
                    "compute_self_delta_pct": 100.0 * (compute_self - base_compute) / max(base_compute, 1e-9),
                    "checker_self_delta_pct": 100.0 * (checker_self - base_checker) / max(base_checker, 1e-9),
                }
            )

    return {"deltas": summary_rows}


def _make_markdown(results: list[dict], summary: dict) -> str:
    lines = []
    lines.append("# Buffer Necessity Benchmark Summary")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("- Modes: `baseline`, `alloc_per_call_checker`, `no_cache_latest`")
    lines.append("- Workloads: `duel_random_random_stable` matrix + `duel_random_vs_random` spot checks")
    lines.append("- Devices: CPU and CUDA (if available)")
    lines.append("")

    lines.append("## Raw Run Highlights")
    lines.append("")
    lines.append("| Device | Script | Board | Games | Plies | Mode | Actions/s | Elapsed(s) |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")
    for row in results:
        lines.append(
            f"| {row['device']} | {row['script']} | {row['board_size']} | {row['num_games']} | "
            f"{row['max_plies']} | {row['mode']} | {row['external_actions_per_s']:.2f} | {row['external_elapsed_s']:.3f} |"
        )
    lines.append("")

    lines.append("## Relative To Baseline")
    lines.append("")
    lines.append("| Device | Script | Board | Games | Plies | Mode | Throughput Δ% | HashSelf Δ% | ComputeSelf Δ% | CheckerSelf Δ% |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in summary["deltas"]:
        lines.append(
            f"| {row['device']} | {row['script']} | {row['board_size']} | {row['num_games']} | "
            f"{row['max_plies']} | {row['mode']} | {row['throughput_delta_pct']:+.2f} | "
            f"{row['hash_self_delta_pct']:+.2f} | {row['compute_self_delta_pct']:+.2f} | "
            f"{row['checker_self_delta_pct']:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    modes = [
        Mode("baseline", {}),
        Mode("alloc_per_call_checker", {"GO_ENGINE_ALLOC_PER_CALL_CHECKER": "1"}),
        Mode("no_cache_latest", {"GO_ENGINE_NO_CACHE_LATEST": "1"}),
    ]

    workloads = [
        # Representative matrix via existing stable duel script.
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 9, 32, 12),
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 9, 64, 12),
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 9, 128, 12),
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 19, 8, 8),
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 19, 16, 8),
        Workload("duel_random_random_stable", duel_stable.simulate_batch_games_with_history, 19, 32, 8),
        # Existing duel script spot checks.
        Workload("duel_random_vs_random", duel_vs_random.simulate_batch_games_with_history, 9, 64, 10),
    ]

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    all_results: list[dict] = []

    for device_kind in devices:
        for workload in workloads:
            seed = (
                workload.board_size * 1_000_000
                + workload.num_games * 1_000
                + workload.max_plies * 10
                + (1 if device_kind == "cuda" else 0)
            )
            for mode in modes:
                row = _run_workload(workload, mode, device_kind, seed=seed)
                all_results.append(row)
                print(
                    f"[done] device={device_kind:>4} script={workload.script_name:<25} "
                    f"H={workload.board_size:<2} B={workload.num_games:<3} T={workload.max_plies:<3} "
                    f"mode={mode.name:<22} actions/s={row['external_actions_per_s']:.2f}"
                )

    summary = _summarize(all_results)

    root = Path(__file__).resolve().parent
    json_path = root / "benchmark_buffer_necessity_results.json"
    md_path = root / "benchmark_buffer_necessity_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2)

    md = _make_markdown(all_results, summary)
    md_path.write_text(md, encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

