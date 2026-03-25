# Benchmark Buffer Necessity Report

This report answers whether pooled workspaces / caches are materially useful.

## 1) Hot Buffer Inventory (Current Code)

### `engine/game_state_machine.py`

| Buffer / Cache | Role | Primary Hot Methods |
|---|---|---|
| `Zpos`, `ZposT` | Zobrist key tables | `_build_candidate_hashes`, `_apply_zobrist_update` |
| `_hash_ws` | Reused `(3, B, N2)` int32 scratch (`placement_delta`, `capture_delta`, `candidate_hashes`) | `_build_candidate_hashes` |
| `_cap_vals` | Reused `(B, N2, 4)` int32 gather/staging for capture XOR | `_build_candidate_hashes` |
| `_group_xor` | Reused `(B*N2,)` int32 scratch for per-group XOR | `_build_candidate_hashes` |
| `_latest_legal_points_raw` | Cached legality mask before ko filter | `legal_points`, `state_transition` via `_get_or_compute_latest` |
| `_latest_legal_info` | Cached CSR payload from checker | `state_transition` via `_get_or_compute_latest` |
| `_latest_candidate_hashes` | Cached candidate hashes for current state | `legal_points`, `state_transition` via `_get_or_compute_latest` |
| `_latest_legal_points` | Cached legality after ko filter | `legal_points` |

### `engine/board_rules_checker.py`

| Buffer / Cache | Role | Primary Hot Methods |
|---|---|---|
| `point_ids`, `neighbor_point_ids`, `neighbor_on_board`, `neighbor_point_ids_safe` | Static neighbor topology tables | `_get_neighbor_colors_batch`, `_get_neighbor_roots_batch`, `_hook_and_compress`, legality/territory methods |
| `_uf_nbr_parent` | Reused `(B, N2, 4)` int32 UF scratch for `take_along_dim(..., out=...)` | `_hook_and_compress` |
| `_csr_sg` | Reused global stone list storage | `_build_group_csr` |
| `_csr_slc`, `_csr_slr` | Reused local LUT buffers `(B, N2)` | `_build_group_csr` |
| `_csr_sp` | Reused CSR pointer storage | `_build_group_csr` |
| `_csr_gptr` | Reused board-group pointer storage | `_build_group_csr` |
| `_csr_capacity_K`, `_csr_capacity_R` | Capacity trackers for pooled CSR buffers | `_build_group_csr` |

## 2) Benchmark Modes

Implemented runtime A/B modes:

- `GO_ENGINE_ALLOC_PER_CALL_HASH=1`
  - Allocates hash scratch buffers per call in `GameStateMachine._build_candidate_hashes`.
- `GO_ENGINE_ALLOC_PER_CALL_CHECKER=1`
  - Allocates UF/CSR checker scratch buffers per call in `BoardRulesChecker`.
- `GO_ENGINE_NO_CACHE_LATEST=1`
  - Disables state-local legality/candidate caching in `GameStateMachine`.

## 3) Workload Matrix

Benchmark harness: [`/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity.py`](/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity.py)

Generated artifacts:

- [`/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity_results.json`](/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity_results.json)
- [`/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity_summary.md`](/workspace/mastering_go_cat_RL_cuda/benchmark_buffer_necessity_summary.md)

Matrix used:

- Modes: `baseline`, `alloc_per_call_hash`, `alloc_per_call_checker`, `no_cache_latest`
- Devices: CPU and CUDA (when available)
- Existing duel workloads:
  - `duel_random_random_stable`
    - `board_size=9`, `num_games={32,64,128}`, `max_plies=12`
    - `board_size=19`, `num_games={8,16,32}`, `max_plies=8`
  - `duel_random_vs_random` spot check
    - `board_size=9`, `num_games=64`, `max_plies=10`
- All runs seeded deterministically per workload/device for A/B fairness.

Total runs executed: 56

Command used:

```bash
python3 benchmark_buffer_necessity.py
```

## 4) Timing + Memory Results

### 4.1 Throughput deltas (vs baseline)

Interpretation uses median deltas from `duel_random_random_stable` workloads, excluding the first small `(board=9, games=32)` case due one-time warmup effects.

| Mode | CPU median throughput delta | CUDA median throughput delta | Interpretation |
|---|---:|---:|---|
| `alloc_per_call_hash` | `-2.57%` | `-0.31%` | no meaningful gain; effectively neutral |
| `alloc_per_call_checker` | `-1.97%` | `-0.15%` | no meaningful gain; effectively neutral |
| `no_cache_latest` | `-26.13%` | `-41.84%` | clear regression |

### 4.2 Method-level timing behavior

Representative CUDA cases (more stable):

- `duel_random_random_stable`, `board=9`, `games=64`, `plies=12`
  - baseline: `_compute_legal_and_candidates` self `25.13 ms`
  - `alloc_per_call_hash`: `25.01 ms` (near-identical)
  - `alloc_per_call_checker`: `25.33 ms` (near-identical)
  - `no_cache_latest`: `49.90 ms` (about 2x)
- `duel_random_random_stable`, `board=19`, `games=32`, `plies=8`
  - baseline: `_compute_legal_and_candidates` self `16.58 ms`
  - `alloc_per_call_hash`: `16.65 ms` (near-identical)
  - `alloc_per_call_checker`: `16.87 ms` (near-identical)
  - `no_cache_latest`: `32.99 ms` (about 2x)

Across the matrix, `no_cache_latest` consistently raised self times for:

- `_build_candidate_hashes`
- `_compute_legal_and_candidates`
- `compute_batch_legal_and_info`

while per-call allocation modes were usually within noise bands.

### 4.3 CUDA memory

Peak allocation differences were small:

| Mode | Median peak alloc (MB) |
|---|---:|
| baseline | `1.10` |
| alloc_per_call_hash | `0.95` |
| alloc_per_call_checker | `1.05` |
| no_cache_latest | `1.10` |

No strong memory-driven reason to prefer per-call allocation modes.

## 5) Keep / Remove / Optional Guidance

### Keep (recommended)

- `Zpos`, `ZposT` (hash correctness + stable semantics)
- Static checker topology tables:
  - `point_ids`
  - `neighbor_point_ids`
  - `neighbor_on_board`
  - `neighbor_point_ids_safe`
- `_latest_*` legality/candidate caches in `GameStateMachine`
  - disabling them (`no_cache_latest`) showed clear regressions

### Optional (performance-neutral in these tests)

- `GameStateMachine` hash scratch pooling:
  - `_hash_ws`, `_cap_vals`, `_group_xor`
- `BoardRulesChecker` pooled UF/CSR buffers:
  - `_uf_nbr_parent`
  - `_csr_sg`, `_csr_slc`, `_csr_slr`, `_csr_sp`, `_csr_gptr`

Measured effect was generally near-neutral when toggled to per-call allocation.
If code simplicity is preferred, these can be simplified behind a config switch.

### Remove now?

Not strongly justified by performance data. Current recommendation:

- Keep defaults as-is (pooled + cached), because they are not hurting and are already integrated.
- Keep benchmark flags for controlled experiments:
  - `GO_ENGINE_ALLOC_PER_CALL_HASH`
  - `GO_ENGINE_ALLOC_PER_CALL_CHECKER`
  - `GO_ENGINE_NO_CACHE_LATEST`

### Practical decision rule

- If simplifying code: remove pooled scratch buffers first, but keep `_latest_*` caches.
- If maximizing throughput consistency: keep current defaults.
- If unsure: keep defaults and re-run `benchmark_buffer_necessity.py` after any refactor.

