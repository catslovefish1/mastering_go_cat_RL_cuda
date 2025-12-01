import time
import torch


def benchmark_allocate_each_iter(B: int, N2: int, iters: int, device: torch.device) -> float:
    """
    Case 1: allocate a fresh (B, N2, 4) tensor every iteration.
    Returns elapsed time in milliseconds.
    """
    print(f"[alloc-each] B={B}, N2={N2}, iters={iters}, device={device}")

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(iters):
            ws = torch.empty((B, N2, 4), dtype=torch.int32, device=device)
            # pretend we do some work
            ws.mul_(2)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        # CPU fallback timing
        start = time.time()
        for _ in range(iters):
            ws = torch.empty((B, N2, 4), dtype=torch.int32, device=device)
            ws.mul_(2)
        elapsed_ms = (time.time() - start) * 1000.0

    return elapsed_ms


def benchmark_reuse_workspace(B: int, N2: int, iters: int, device: torch.device) -> float:
    """
    Case 2: allocate a single (B, N2, 4) tensor once, reuse it every iteration.
    Returns elapsed time in milliseconds.
    """
    print(f"[reuse-ws]  B={B}, N2={N2}, iters={iters}, device={device}")

    # single allocation
    ws = torch.empty((B, N2, 4), dtype=torch.int32, device=device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(iters):
            # reuse the same buffer – e.g. clear or overwrite
            ws.mul_(2)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start = time.time()
        for _ in range(iters):
            ws.mul_(2)
        elapsed_ms = (time.time() - start) * 1000.0

    return elapsed_ms


def main():
    # ---- config ----
    B = 1024          # batch size
    N = 19           # board size
    N2 = N * N
    iters = 5000      # adjust up/down depending on speed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Warmup (especially for CUDA)
    if device.type == "cuda":
        _ = torch.empty((B, N2, 4), dtype=torch.int32, device=device)
        torch.cuda.synchronize()

    # ---- run benchmarks ----
    t_alloc = benchmark_allocate_each_iter(B, N2, iters, device)
    t_reuse = benchmark_reuse_workspace(B, N2, iters, device)

    print()
    print("==== Results ====")
    print(f"Allocate each iter: {t_alloc:.2f} ms total  ({t_alloc / iters:.4f} ms / iter)")
    print(f"Reuse workspace  : {t_reuse:.2f} ms total  ({t_reuse / iters:.4f} ms / iter)")
    if t_reuse > 0:
        print(f"Speedup (alloc-each / reuse) ≈ {t_alloc / t_reuse:.2f}x")


if __name__ == "__main__":
    main()
