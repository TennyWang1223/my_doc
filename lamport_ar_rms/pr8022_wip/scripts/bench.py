"""Lamport vs Gluon fused AR + RMSNorm (+ per-token FP8), TP4, bf16.

Times each candidate in a graph of CALLS calls, median of REPLAYS, slowest rank.
Also bit-compares lamport vs gluon outputs (informational).
Usage: python bench.py -m 1 2 4 ... -n 2048 ... -q 0 1 [--rounds R]
"""

import argparse
import itertools
import os
import statistics
import sys
from multiprocessing import Pool, set_start_method

os.environ.setdefault("AITER_GLUON_FUSED_AR", "1")

import torch
import torch.distributed as dist

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
CALLS = int(os.environ.get("BENCH_CALLS", 10))
REPLAYS = int(os.environ.get("BENCH_REPLAYS", 20))


def time_graph(fn):
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        fn()
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(CALLS):
                fn()
    torch.cuda.synchronize()
    get_tp_group().barrier()
    graph.replay()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(REPLAYS)
    ]
    for s, e in ev:
        s.record()
        graph.replay()
        e.record()
    torch.cuda.synchronize()
    us = statistics.median(s.elapsed_time(e) * 1000 / CALLS for s, e in ev)
    graph.reset()
    return us


def bitdiff(a, b):
    a = a.contiguous().view(torch.uint8)
    b = b.contiguous().view(torch.uint8)
    return int((a != b).sum().item())


def worker(tp, rank, ms, ns, quants, cands_sel, rounds, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    comm = get_tp_group().device_communicator
    ca = comm.ca_comm
    from aiter.dist.device_communicators.communicator_cuda import _gluon_ar_norm

    rows = []
    for rnd in range(rounds):
        for quant, n, m in itertools.product(quants, ns, ms):
            g = torch.Generator(device="cuda").manual_seed(n * 1000 + m + rank)
            x = torch.randn((m, n), device="cuda", generator=g).to(torch.bfloat16)
            res = torch.randn((m, n), device="cuda", generator=g).to(torch.bfloat16)
            w = torch.randn((n,), device="cuda").to(torch.bfloat16)
            dist.broadcast(w, 0, group=get_tp_group().device_group)
            c = {}
            if ca.should_lamport_ar_rms(x, res, w, n):
                if quant:
                    c["lamport"] = lambda: ca.lamport_fused_ar_rms_quant(x, res, w, EPS)
                else:
                    c["lamport"] = lambda: ca.lamport_fused_ar_rms(x, res, w, EPS)
            if comm.gluon_ar_plain is not None:
                c["gluon"] = lambda: _gluon_ar_norm(
                    comm, x, res, w, EPS, False, quant=bool(quant)
                )
            if ca.should_lamport_ar_add(x, res):
                c["add"] = lambda: ca.lamport_ar_add(x, res)
            c = {k: v for k, v in c.items() if not cands_sel or k in cands_sel}
            r = {"m": m, "n": n, "q": quant, "round": rnd}
            if "lamport" in c and "gluon" in c and rnd == 0:
                a, b = c["lamport"](), c["gluon"]()
                r["diff"] = max(bitdiff(a[0], b[0]), bitdiff(a[1], b[1]))
            for k, fn in c.items():
                r[k] = time_graph(fn)
            rows.append(r)
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t", type=int, default=4)
    p.add_argument("-m", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("-n", type=int, nargs="*", default=[2048, 4096, 6144, 7168, 8192])
    p.add_argument("-q", type=int, nargs="*", default=[0, 1])
    p.add_argument("-c", nargs="*", default=["lamport", "gluon"])
    p.add_argument("--rounds", type=int, default=1)
    a = p.parse_args()
    set_start_method("spawn", force=True)
    im = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=a.t) as pool:
        jobs = [
            pool.apply_async(worker, (a.t, r, a.m, a.n, a.q, a.c, a.rounds, im))
            for r in range(a.t)
        ]
        res = [j.get(timeout=3600) for j in jobs]
    # slowest rank per row, then median over rounds
    merged = {}
    for cells in zip(*res):
        r = dict(cells[0])
        for k in ("lamport", "gluon", "diff", "add"):
            if k in r:
                r[k] = max(cc[k] for cc in cells)
        merged.setdefault((r["q"], r["n"], r["m"]), []).append(r)
    print("| q | n | m | KB | lamport us | gluon us | gluon/lamport | diff |")
    print("|---|---|---|---|---|---|---|---|")
    for (q, n, m), rs in merged.items():
        lam = [r["lamport"] for r in rs if "lamport" in r]
        glu = [r["gluon"] for r in rs if "gluon" in r]
        lm = statistics.median(lam) if lam else float("nan")
        gm = statistics.median(glu) if glu else float("nan")
        d = rs[0].get("diff", "")
        ad = [r["add"] for r in rs if "add" in r]
        am = statistics.median(ad) if ad else float("nan")
        print(
            f"| {q} | {n} | {m} | {m*n*2/1024:.0f} | {lm:.2f} | {gm:.2f} | {gm/lm:.3f} | {d} | {am:.2f} |"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
