# Buffer Necessity Benchmark Summary

## Matrix

- Modes: `baseline`, `alloc_per_call_hash`, `alloc_per_call_checker`, `no_cache_latest`
- Workloads: `duel_random_random_stable` matrix + `duel_random_vs_random` spot checks
- Devices: CPU and CUDA (if available)

## Raw Run Highlights

| Device | Script | Board | Games | Plies | Mode | Actions/s | Elapsed(s) |
|---|---|---:|---:|---:|---|---:|---:|
| cpu | duel_random_random_stable | 9 | 32 | 12 | baseline | 9500.53 | 0.040 |
| cpu | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_hash | 9803.17 | 0.039 |
| cpu | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_checker | 4237.46 | 0.091 |
| cpu | duel_random_random_stable | 9 | 32 | 12 | no_cache_latest | 3106.47 | 0.124 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | baseline | 4756.28 | 0.161 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_hash | 7256.74 | 0.106 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_checker | 6851.97 | 0.112 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | no_cache_latest | 3744.36 | 0.205 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | baseline | 8335.66 | 0.184 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_hash | 8121.69 | 0.189 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_checker | 11323.86 | 0.136 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | no_cache_latest | 5084.09 | 0.302 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | baseline | 858.83 | 0.075 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_hash | 3044.19 | 0.021 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_checker | 841.87 | 0.076 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | no_cache_latest | 705.78 | 0.091 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | baseline | 4405.30 | 0.029 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_hash | 1509.24 | 0.085 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_checker | 1520.71 | 0.084 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | no_cache_latest | 1195.86 | 0.107 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | baseline | 2735.17 | 0.094 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_hash | 2630.93 | 0.097 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_checker | 2636.64 | 0.097 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | no_cache_latest | 2020.55 | 0.127 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | baseline | 5919.20 | 0.108 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_hash | 3936.35 | 0.163 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_checker | 6073.86 | 0.105 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | no_cache_latest | 4599.94 | 0.139 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | baseline | 1107.20 | 0.347 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_hash | 7575.75 | 0.051 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_checker | 7597.22 | 0.051 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | no_cache_latest | 4372.21 | 0.088 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | baseline | 14825.90 | 0.052 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_hash | 14762.73 | 0.052 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_checker | 14803.54 | 0.052 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | no_cache_latest | 8537.55 | 0.090 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | baseline | 24131.27 | 0.064 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_hash | 29379.24 | 0.052 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_checker | 29302.05 | 0.052 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | no_cache_latest | 16516.87 | 0.093 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | baseline | 1933.11 | 0.033 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_hash | 1927.08 | 0.033 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_checker | 1923.13 | 0.033 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | no_cache_latest | 1124.28 | 0.057 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | baseline | 3754.01 | 0.034 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_hash | 3812.96 | 0.034 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_checker | 3821.45 | 0.033 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | no_cache_latest | 2232.04 | 0.057 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | baseline | 7611.84 | 0.034 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_hash | 7507.57 | 0.034 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_checker | 7549.84 | 0.034 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | no_cache_latest | 4407.03 | 0.058 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | baseline | 8820.41 | 0.073 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_hash | 13333.29 | 0.048 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_checker | 13361.71 | 0.048 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | no_cache_latest | 8098.63 | 0.079 |

## Relative To Baseline

| Device | Script | Board | Games | Plies | Mode | Throughput Δ% | HashSelf Δ% | ComputeSelf Δ% | CheckerSelf Δ% |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| cpu | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_hash | +3.19 | +52.81 | -8.82 | -8.85 |
| cpu | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_checker | -55.40 | -1.31 | +199.85 | -9.51 |
| cpu | duel_random_random_stable | 9 | 32 | 12 | no_cache_latest | -67.30 | +96.82 | +305.93 | +84.26 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_hash | +52.57 | +2.33 | -38.15 | -5.90 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_checker | +44.06 | +964.06 | -73.87 | -1.54 |
| cpu | duel_random_random_stable | 9 | 64 | 12 | no_cache_latest | -21.28 | +100.50 | +25.63 | +91.28 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_hash | -2.57 | +1.99 | -30.36 | +870.00 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_checker | +35.85 | -1.99 | -30.16 | +17.85 |
| cpu | duel_random_random_stable | 9 | 128 | 12 | no_cache_latest | -39.01 | +96.27 | +66.96 | +96.31 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_hash | +254.46 | +3.20 | -80.63 | -2.58 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_checker | -1.97 | -0.36 | +2.29 | -4.12 |
| cpu | duel_random_random_stable | 19 | 8 | 8 | no_cache_latest | -17.82 | +2017.08 | -61.22 | +92.27 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_hash | -65.74 | +1.80 | +295.50 | +0.00 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_checker | -65.48 | +0.60 | +292.11 | -2.31 |
| cpu | duel_random_random_stable | 19 | 16 | 8 | no_cache_latest | -72.85 | +112.91 | +391.37 | +108.08 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_hash | -3.81 | +1367.73 | -65.95 | +7.82 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_checker | -3.60 | +0.73 | -64.09 | +21.76 |
| cpu | duel_random_random_stable | 19 | 32 | 8 | no_cache_latest | -26.13 | +101.22 | +34.15 | +100.00 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_hash | -33.50 | -0.56 | +1.18 | -16.25 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_checker | +2.61 | -2.96 | -66.11 | -16.53 |
| cpu | duel_random_vs_random | 9 | 64 | 10 | no_cache_latest | -22.29 | +1105.93 | -32.39 | +68.32 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_hash | +584.23 | -49.33 | -86.10 | -7.24 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | alloc_per_call_checker | +586.17 | -50.79 | -85.92 | -8.14 |
| cuda | duel_random_random_stable | 9 | 32 | 12 | no_cache_latest | +294.89 | -1.50 | -72.23 | +83.03 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_hash | -0.43 | +3.31 | -0.55 | +0.74 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | alloc_per_call_checker | -0.15 | -0.38 | +0.59 | -0.25 |
| cuda | duel_random_random_stable | 9 | 64 | 12 | no_cache_latest | -42.41 | +99.32 | +97.74 | +97.78 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_hash | +21.75 | +2.27 | -31.25 | -4.94 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | alloc_per_call_checker | +21.43 | -0.22 | -30.08 | -5.41 |
| cuda | duel_random_random_stable | 9 | 128 | 12 | no_cache_latest | -31.55 | +105.20 | +38.83 | +94.35 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_hash | -0.31 | +3.43 | -0.42 | -1.47 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | alloc_per_call_checker | -0.52 | +0.27 | +1.09 | -0.74 |
| cuda | duel_random_random_stable | 19 | 8 | 8 | no_cache_latest | -41.84 | +100.55 | +99.09 | +99.63 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_hash | +1.57 | +3.07 | +0.67 | +1.10 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | alloc_per_call_checker | +1.80 | -0.13 | +1.46 | -1.10 |
| cuda | duel_random_random_stable | 19 | 16 | 8 | no_cache_latest | -40.54 | +99.47 | +100.24 | +98.17 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_hash | -1.37 | +3.65 | -0.06 | -0.35 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | alloc_per_call_checker | -0.81 | +0.26 | +1.38 | -0.70 |
| cuda | duel_random_random_stable | 19 | 32 | 8 | no_cache_latest | -42.10 | +100.52 | +99.58 | +96.14 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_hash | +51.16 | +2.87 | +0.62 | +0.88 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | alloc_per_call_checker | +51.49 | -1.05 | +1.25 | -1.47 |
| cuda | duel_random_vs_random | 9 | 64 | 10 | no_cache_latest | -8.18 | +99.62 | +100.19 | +98.83 |
