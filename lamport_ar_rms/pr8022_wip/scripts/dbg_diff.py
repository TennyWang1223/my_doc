"""Locate bit differences of lamport_fused_ar_rms vs car_fused (TP4)."""

import os
import sys
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

EPS = 1e-6


def worker(tp, rank, m, n, reps, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method=init_method)
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    g = torch.Generator(device="cuda").manual_seed(n * 1000 + m)
    xs = torch.randn((tp, m, n), device="cuda", generator=g).to(torch.bfloat16)
    res = torch.randn((m, n), device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn((n,), device="cuda", generator=g).to(torch.bfloat16)
    x = xs[rank].contiguous()
    out = []
    base = ca.custom_fused_ar_rms(x, res, w, EPS, use_1stage=True)
    for r in range(reps):
        got = ca.lamport_fused_ar_rms(x, res, w, EPS)
        for name, a, b in (("out", got[0], base[0]), ("res_out", got[1], base[1])):
            d = (a.view(torch.int16) != b.view(torch.int16)).nonzero()
            if d.numel():
                rows = sorted(set(d[:, 0].tolist()))
                out.append((r, name, d.shape[0], rows[:10], d[:6].tolist()))
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return out


if __name__ == "__main__":
    m, n, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    set_start_method("spawn", force=True)
    im = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(4) as pool:
        jobs = [pool.apply_async(worker, (4, r, m, n, reps, im)) for r in range(4)]
        for r, j in enumerate(jobs):
            print("rank", r, j.get(timeout=600))
