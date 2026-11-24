"""Per-wave phase timestamps of lamport_fused_ar_rms (AITER_LAMPORT_RMS_DBG=16).
Prints, per rank, the phase times (us) relative to the earliest t0 of row 0:
t0 start, t1 sends issued, t2 poll done, t3 reduce done, t4 outputs stored,
t5 all stores drained (incl. rearm and remote sends)."""

import os
import sys
from multiprocessing import Pool, set_start_method

import torch

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

EPS = 1e-6
CALLS = 10


def worker(tp, rank, m, n, q, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method=init_method)
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    x = torch.randn((m, n), device="cuda").to(torch.bfloat16)
    res = torch.randn((m, n), device="cuda").to(torch.bfloat16)
    w = torch.randn((n,), device="cuda").to(torch.bfloat16)
    fn = (lambda: ca.lamport_fused_ar_rms_quant(x, res, w, EPS)) if q else (
        lambda: ca.lamport_fused_ar_rms(x, res, w, EPS))
    graph = torch.cuda.CUDAGraph()
    outs = []
    with graph_capture() as gc:
        fn()
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(CALLS):
                outs.append(fn())
    torch.cuda.synchronize()
    get_tp_group().barrier()
    res_all = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        calls = []
        for o in outs:
            r = o[1][0].contiguous().view(torch.int64).cpu()
            waves = r.numel() // 64
            ts = r[: waves * 64].view(waves, 64)[:, :6]
            ts = ts[(ts[:, 0] > 0)]
            calls.append(ts)
        res_all.append(calls)
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return res_all


if __name__ == "__main__":
    m, n, q = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    set_start_method("spawn", force=True)
    im = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(4) as pool:
        jobs = [pool.apply_async(worker, (4, r, m, n, q, im)) for r in range(4)]
        res = [j.get(timeout=600) for j in jobs]
    # per call: phases relative to each rank's own earliest t0 (10 ns ticks)
    import statistics

    phases = {i: [] for i in range(1, 6)}
    spans = []
    for rank in range(4):
        for rep in res[rank][1:]:
            for ts in rep[1:]:
                t0 = ts[:, 0].min().item()
                for i in range(1, 6):
                    phases[i].append((ts[:, i].max().item() - t0) / 100)
                spans.append((ts[:, 0].max().item() - t0) / 100)
    print("waves", res[0][1][1].shape[0], "start spread us", f"{statistics.median(spans):.2f}")
    names = {1: "pre-wait", 2: "poll done", 3: "reduce done", 4: "end issued", 5: "stores drained"}
    for i in range(1, 6):
        v = phases[i]
        print(f"{names[i]:15s} median {statistics.median(v):.2f} us  p10 {sorted(v)[len(v)//10]:.2f}  p90 {sorted(v)[len(v)*9//10]:.2f}")
