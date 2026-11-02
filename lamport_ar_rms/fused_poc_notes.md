# allreduce fuse rmsnorm 调研 + Lamport one-shot 融合 POC 记录

> - **整理日期**：2026-10-29
> - **代码位置**：`/mnt/raid0/tiwang/work/review_code/demo_1019/aiter`，基于本地 main `84077b3c0`，所有改动都没有 commit
> - **机器**：8 × MI308X（gfx942），xGMI。每卡 80 CU、4 个 XCD（`num_xcc 4`）、每个 SIMD 最多 8 个 wave
> - **相关文档**：
>   - `LAMPORT_ONESHOT_PROGRESS.md`：纯 Lamport one-shot AR 的 POC 进度
>   - `LAMPORT_ONESHOT_DESIGN.md`：Lamport 移植到 gfx1250 的指南
>   - 本文：fused AR + RMSNorm 的调研、讨论、融合 POC 和测试结果

## 0. 下次接着做：先看这里

**当前状态**：Lamport one-shot AR + residual add + RMSNorm 的融合 POC 已经跑通，接口独立。
- **正确性**：out 和 residual_out 都和现有 fused 1-stage kernel 逐位一致。唯一例外是 resize 检查里的 1 个元素，见 5.3 节。
- **性能**：已经和现有 fused 做了对比，结论见第 5 节。
- **小尺寸**：TP2 全区间快 9–26%，TP4 64KB 以内快 0–17%，TP8 64KB 以内快 7–25%。
- **大尺寸**：持平或变慢，这是 one-shot 本身的问题。

**最值得做的下一步**：小 M 时，融合版比纯 Lamport AR 多 2–3.5 µs，这部分开销还没拆清楚，见 6.1 节。

**切分 H 和性能的关系**（2026-10-30 讨论）：整理在同目录的 `split_h_and_cu_vmem_limits.md`，先看它的 §0。里面先讲了硬件基础（单个 CU 的两道关），时间已按实测的 1.85 GHz 更正。

**跑测试之前**：
```bash
cd /mnt/raid0/tiwang/work/review_code/demo_1019/aiter
# 必须设置 PYTHONPATH：环境里的 editable aiter 指向 review_code/main/aiter，不是这个目录
PYTHONPATH=$PWD python3 op_tests/multigpu_tests/test_lamport_oneshot_rmsnorm.py            # 全量：TP 2/4/8，H 7168/8192，M = 1..32
PYTHONPATH=$PWD python3 op_tests/multigpu_tests/test_lamport_oneshot_rmsnorm.py -t 8 -n 7168 -m 1 8
# 改了 .cuh 一定要先删 JIT 产物，否则测的还是旧代码
rm -f aiter/jit/module_custom_all_reduce.so && rm -rf aiter/jit/build/module_custom_all_reduce
```

---

## 1. 调研：aiter 里 allreduce fuse rmsnorm 的现状（本地 main 84077b3c0）

### 1.1 quant 变体

都在老模块 `module_custom_all_reduce` 里：

| 变体 | op | 引入 | 输出 | kernel 路径 | 后加的选项 | 限制 |
|---|---|---|---|---|---|---|
| per-token FP8 | `fused_allreduce_rmsnorm_quant` | #1990 Yutao Xu 2026-02-11 | fp8 + scale `(M,1)` | 1stage / 2stage(≤512KB) / split | `gemma_norm`(#3981)、`emit_bf16`(#4110) | 不支持 pad。communicator 只在 hidden ∈ {512,1024,2048,4096} 或 fp16/bf16 的 {6144,7168}、且 ≤4MB 时融合，否则是基础版加单独的 HIP per-token quant |
| per-group FP8 | `fused_allreduce_rmsnorm_quant_per_group` | #2823 Hubert Lu 2026-04-28（同一个 PR 带了 bf16_out） | fp8 + scale `(M,K/g)` | 1stage / 2stage / split | `group_size`(默认 128)、`transpose_scale`(#3652/#4478)、`gemma_norm`(#6472) | 只支持 fp16/bf16；group_size 有 5 条约束 |
| MXFP4 | `fused_allreduce_rmsnorm_mxfp4_quant` | #3229 Hubert Lu 2026-06-02 | fp4 `(M,K/2)` + e8m0 `(M,K/32)` | 只有 1stage / 2stage | `bf16_out`；`gemma_norm` 由 #6000 在 10-19 合入（本地 84077b3c0 里还没有） | K%32==0；不满足时 fallback 到基础版 + Triton `dynamic_mxfp4_quant` |

- **统一入口**：`tensor_model_parallel_fused_allreduce_rmsnorm_quant(quant_type=...)`，由 #4039（2026-07-02）统一。
- **不带 quant 的同类变体**：
  - `fused_allreduce_rmsnorm_pad`（#3265 PerryZhang01）
  - QuickReduce 的 `qr_all_reduce_rmsnorm`（#4104 Mikko Tukiainen，只支持 gfx94x/95x）
  - `fused_qknorm_allreduce(_rope)`（MiniMax，#3163/#3650）
  - `fused_allreduce_mhc_post_*`（#3791，不带 norm）
  - Triton/Iris 的 `reduce_scatter_rmsnorm_quant_all_gather`
  - FlyDSL 版（Ville Pietilä，PR #5670 draft，只支持 gfx942/950、bf16，需要 `AITER_FLY_AR=1`）

### 1.2 基础版的参数演进

你最早的版本是 #1290（2025-10-31）：
- op：`fused_allreduce_rmsnorm(_fa, inp, out, w, eps, reg_buffer=None)`
- 顶层：`tensor_model_parallel_fused_allreduce_rmsnorm(input_, weight_, eps)`
- kernel：`reduce_scatter_cross_device_store` + `local_device_load_rmsnorm[_naive]`，两个 kernel

现在：
```python
tensor_model_parallel_fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps,
    prefill_support=False, x_pad_to_multiple=0, gemma_norm=False) -> (out, residual_out)
fused_allreduce_rmsnorm(_fa, inp, res_inp, res_out, out, w, eps, reg_ptr, reg_bytes, use_1stage, gemma_norm=False)
```

| 参数 | 引入 | 作用 |
|---|---|---|
| `residual_inp_` / `res_inp, res_out` | #1401 TennyWang 2025-11-14 | 融合 residual add |
| `use_1stage` | #1949 Yutao Xu 2026-02-05 | 单 kernel 的 `allreduce_fusion_kernel_1stage`；为 false 时还是两个 kernel |
| `reg_buffer` → `reg_ptr, reg_bytes` | #2453 TennyWang 2026-04-01 | C++ 去 torch 化 |
| `prefill_support` | #2453 | 放开 prefill 尺寸 |
| `x_pad_to_multiple`（C++ 里是 `input_hidden_dim`/`out_n`），以及 `_pad` op | #3265 PerryZhang01 2026-06-02 | padding |
| `gemma_norm` | #3792 Zhu Yuhua 2026-06-18 | `x*(1+w)` |

不在参数里、但影响行为的：
- **`use_1stage` 自动选择**：`AITER_AR_1STAGE` 强制开关；`AITER_AR_1STAGE_MAX_KB` 默认是 `128*7168*2/world_size` 字节。注意这个变量名叫 KB，但从 #3745 起代码一直把它当字节用。
- **融合门槛**：n ≤ 16384、总量 < 64MB、TP ≠ 6。
- **其他**：非连续输入会用 `hipMemcpy2DAsync` 打包；n 不对齐的情况（比如 2880）由 #2329 支持；基础版只有 1stage 和 split 两条路，2stage 只给 quant 变体用。

### 1.3 顺带发现的小问题（本地 84077b3c0）

- `AITER_AR_1STAGE_MAX_KB` 的单位和名字不符，实际是字节。
- per-token quant 的融合分支（`communicator_cuda.py` 的 `fused_allreduce_rmsnorm_quant`）缺少保护：没有排除 TP6（C++ 会直接 throw）；`ca_comm` 为 None 或 `custom_fused_ar_rms_quant` 返回 None 时，解包会报错。per-group 和 MXFP4 两条路有这些保护。

---

## 2. 调研：gfx1250 上的 allreduce fuse rmsnorm（截至 2026-10-29）

**注意**：本地 main 落后 origin/main 263 个 commit（origin/main = `269675dea`），下面这些都要 `git pull` 才能看到。

### 2.1 已合入 main

- **#6699** XiaobingSuper（10-28），gfx1250 自己的 HIP fused AR + RMSNorm 和 per-token FP8：
  - **one-stage**：direct-read（pull）。
  - **two-stage**：条件是 `!use_1stage`、TP≥4、hidden≥6144、`hidden/8 % ws==0`、shard ≤ 16 MiB、总量 > 4 MiB。每个 rank 的 meta buffer 因此多出 16 MiB staging。
  - **LL 范围（≤4 MiB）**：走 LL AR + 单独 RMSNorm，不融合。作者试过基于 LL 协议的融合 kernel，64 行以下打不过拆开的写法，原因是整行要放在一个 block 里，参与轮询的 CU 太少。
  - **per-token FP8 门槛**：gfx1250 上的默认值放宽到 64 MiB。
  - **修了老模块拿 gfx1250 句柄跑的 bug**：MiniMax-M3 TP4 + Eagle3 出现错 token。修完后 per-group 和 MXFP4 在 gfx1250 上走不融合的路径，qknorm 直接报错。
  - **验证范围**：只在 MI455X 上测了 TP2/TP4；TP8 没验证；CI 没有 gfx1250 runner。
- **#6923**（#6924 叠在它上面一起合入）和 **#6974**，Boss2002n（Satya Nikhil Kodukula）：
  - Gluon 单次 launch 完成 AR + residual + RMSNorm（kernel 里可选 fp8 group-128 quant）。
  - 只支持 TP4、bf16、hidden 为 2 的幂且 ≥256、≤8 MiB。
  - 需要设置 `AITER_GLUON_FUSED_AR=1`。
  - dist 层调用时传的是 `quant=False`，quant 版本没有接进来。
- **#6756**：TennyWang，gfx1250 LL AR 的修复。
- **#5965**：Hubert Lu，在 gfx1250 上开了 QuickReduce，但只支持 INT4 PullQ4，QR 的 fused RMSNorm 调用会直接 throw。

**gfx1250 上调 `fused_allreduce_rmsnorm` 的实际路径**，按优先级：
1. Gluon（需要设环境变量，且只支持 TP4）
2. ≤4 MiB：LL AR + 单独 RMSNorm
3. >4 MiB 且连续、无 pad：HIP fused（one-stage 或 two-stage）
4. per-group / MXFP4：不融合

### 2.2 还没合入的 PR

- **#7082** xytpai：LL 范围内，把 AR 之后的 residual + RMSNorm + per-token FP8 从两个 kernel 合成一个，LL AR 本身不变。
  - 新增 `add_rmsnorm_quant(..., round_before_quant=True)`：先把 norm 结果舍入成 bf16，再求 amax 和量化，这样和拆开的路径逐位一致。
  - 全零行的 scale 规则照搬单独的 quant kernel：一般宽度下 amax 从 0 起算；宽度为 32/64/128 时保留 `kAmaxFloor`=1e-10。
  - 生效条件：per_token、`emit_bf16=False`、QR 关闭、hidden%32==0 且 ≤8192。
  - 只测了 epilogue：1.6–1.7x（1 行时 3.49 → 2.01 µs）。
- **#7080** xytpai：把 two-stage FP8 的 amax 归约改成并行，大 shape 约快 5%。
- **#7122** Boss2002n（draft）：Gluon 的 GEMM + AR + RMSNorm（Qwen3.5 o_proj），1 token 时 6.34 µs，对比拆开的 7.23 µs。
- **#7064** xytpai：wave32 下单独 RMSNorm 的优化，和 AR 无关。

### 2.3 gfx1250 上还没人做的

- per-group FP8 / MXFP4 的 fused 版本
- qknorm AR、MHC post
- pad 或 strided 输入
- TP8 验证
- QR + RMSNorm
- LL 范围内真正的单 kernel AR + RMSNorm（目前只有 Gluon 的 TP4 版本）

---

## 3. 设计讨论的结论

### 3.1 push + 1-stage 能不能把 memory 和 compute 排成流水

- **1-stage 是 latency-bound**：reduce + norm 的计算是亚微秒级，按时间排流水收益很小。
  - #6699 的数据：barrier 同步的 fused 版本 1 行要 19.6 µs，LL AR + RMSNorm 是 11.5 µs。
  - #6924 的数据：Gluon 5.05 µs，aiter fused CAR 8.57 µs。
  - 也就是说，协议选择和行到 CU 的映射能拉开 1.7–2 倍，计算 overlap 最多省几百 ns。
- **"不需要 overlap"这个矛盾，本身就划出了 1-stage 的边界**：数据大到需要 overlap 时，one-shot 要推 (N−1)×S，2-shot 只要 2(N−1)/N×S，TP8 下差 4 倍，这时就该换 2-stage。
- **真正有用的是空间上的分工和细粒度的到达检测**：
  - flag-in-data 或 sentinel，去掉 barrier
  - sender 和 receiver 分在不同 CU 上
  - 等远端数据之前先把 residual 和 weight load 好（Gluon 就是这么做的）
  - 一次把所有 load 发出去，交给硬件去重叠
- **push 的额外好处**：不需要 copy-in，只有目的端（inbox）需要注册。gfx1250 上每次 CAR 都要 copy-in；#6699 给 LL AR 去掉 copy-in 后，1 行从 10.4 µs 降到 8.8 µs。
- **确定性**：必须按固定的 rank 顺序求和（参考 #6954），不能按到达顺序累加，partial 也不能用 float atomicAdd。这并不妨碍流水：load 可以全部先发，只是消费时按顺序。
- **加 quant 时的一个技巧**：amax(归一化后) = inv_rms × max|x·w|，所以 sumsq 和 max 可以在同一次归约里算完。代价是和拆开的路径不再逐位一致。
- **更大的 overlap 在 kernel 之间**：把 push 放进 GEMM 的 epilogue，参考 #7122。

### 3.2 MI400 workgroup cluster

出处：`/home/tennwang/gpu-docs/mi400/spi/mi400_workgroup_clusters_text.txt`

- 一个 cluster 最多 16 个 WG（MI400）。
- 同一个 cluster 的 WG 不会落在同一个 CU 上。
- cluster 里所有 WG 都分到资源后才一起启动，所以 WG 之间互相等待不会死锁。
- 有硬件 cluster barrier。
- WG 之间通过 LDS 直接传数据还不支持（文档原文写的是 "someday"）。
- 每个 SE 同时最多有 15 个 cluster 在跑。
- cluster size = 1 时，barrier 会被硬件变成 NOP。

### 3.3 M 和 H，以及"什么时候切行"（纠正过一次表述）

- **M 和 H**：M 是行数，也就是 token 数（dim0）；H 是 hidden（dim1），RMSNorm 在 H 上做。
- **切行要同时满足两个条件**：
  1. 一行的活够多：每行要推和收 (N−1)×H×2B，大到一个 WG 处理不快；
  2. M 太小，一行一个 WG 铺不满 GPU。
- **M 小但 H 也小时，切了没意义**。
- **H 大到一个 WG 装不下的情况，在 LLM 里基本碰不到**：1024 线程 × 16 个 bf16 就是 16K。
- **证据**：你的纯 Lamport 数据，1×7168，从 bs64（14 个 block）到 bs512（2 个 block）：TP2 下 5.64 → 5.82 µs，TP8 下 7.20 → 9.18 µs。同样的 H，在 TP8 下才算"一行的活多"。
- **实现上**：固定每个 WG 的 chunk，k = H/(8·BS) 由 H 决定；M 只用来决定要不要退回 k=1。

### 3.4 为什么选 Lamport

- sentinel 代替 flag，有效载荷 100%。对比之下 gfx1250 的 LL 是 50%，Gluon 是 75%。而且是 push。
- 现有 fused 1-stage 是 pull：`start_sync`，然后每个 block 直接读所有 rank 的输入，再 `end_sync`。
- 代价：rearm（每个 pack 每轮 N−1 次本地 16B 写）和 3 份 buffer。
- 你的纯 AR 数据显示 one-shot 只在一段窗口内赢：TP2 全部、TP4 28–128KB、TP8 28–64KB。

---

## 4. 融合 POC 的实现

### 4.1 改动的文件

都没有 commit，和原来的 Lamport POC 改在同一批文件里，原来的部分没动：

| 文件 | 内容 |
|---|---|
| `csrc/include/custom_all_reduce.cuh` | 加了 `#include <memory>`；`allreduce_fusion_kernel_1stage` 后面加了常量 `kOneShotRmsMaxGroups`、`oneshot_rmsnorm_buffer_bytes()`、kernel `lamport_allreduce_rmsnorm_oneshot<T, ngpus, BLOCK_SIZE>`（约第 2466 行）、`struct OneShotRmsState`（约第 2658 行，含 `run<T>()` 的 launch 分发）；`CustomAllreduce` 里加了成员 `oneshot_rms_states_` 和 `init_oneshot_rmsnorm()`（约第 5345 行） |
| `csrc/kernels/custom_all_reduce.cu` | 入口 `oneshot_rmsnorm_buffer_size`、`init_oneshot_rmsnorm`、`all_reduce_rmsnorm_oneshot`（约第 160 行） |
| `csrc/include/custom_all_reduce.h` | 声明 |
| `csrc/include/rocm_ops.hpp` | `CUSTOM_ALL_REDUCE_PYBIND` 里加了 3 个 `m.def` |
| `aiter/ops/custom_all_reduce.py` | 3 个 `compile_ops` stub |
| `op_tests/multigpu_tests/test_lamport_oneshot_rmsnorm.py` | 新增测试 |

### 4.2 接口

```python
size  = aiter.oneshot_rmsnorm_buffer_size(world_size)        # inbox + partial slots
state = aiter.init_oneshot_rmsnorm(ca._ptr, self_ptr, handle_ptrs, offsets, threads, hidden)  # 返回 int64 state
aiter.all_reduce_rmsnorm_oneshot(state, inp, res_inp, res_out, out, w, eps, gemma_norm=False)
```
- **buffer**：由调用方用 `ca._pool.create(key, size, uncached=True)` 分配，向上取整到 2MB，再交换 IPC handle。
- **init 之后**：必须先 `torch.cuda.synchronize()`，再做一次跨 rank 的 barrier，原因和 `init_oneshot` 一样：init 会把 buffer memset 成 0xff。
- **生命周期**：state 归 `CustomAllreduce` 所有（`unique_ptr`），随 CA 一起析构；IPC handle 走 CA 的缓存，也由 CA 关闭。
- **限制**：
  - 只支持 2 字节 dtype（bf16/fp16）。
  - TP 只支持 2、4、8。
  - threads ∈ {64, 128, 256, 512, 896, 1024}，并且要满足 `(hidden/8) % threads == 0` 和 `hidden/8 ≤ 1024`。
  - 单次最大 640KB（`kOneShotMaxPacks`）：H=7168 时最多 45 行，H=8192 时最多 40 行。
  - 张量必须连续。

### 4.3 kernel 设计（`lamport_allreduce_rmsnorm_oneshot`）

1. **通信**：完全照搬 `lamport_allreduce_oneshot`。
   - 每个线程负责一个 16B pack：canonicalize 之后，nt store 到每个 peer 的 `slot[rank]`。
   - 轮询本地 inbox，直到没有 0xFFFF sentinel：n 条 nt load 一次全部发出，再用 mask 判断；轮询循环里带 `asm volatile("" ::: "memory")`。
   - 按 rank 0..n−1 的顺序做 fp32 求和。
   - rearm `(round+2)%3`；每个 block 一个 round 计数器。
2. **本地操作数提前取**：远端 store 之后、轮询之前，先 load residual 和 weight。
3. **epilogue 按 `allreduce_fusion_kernel_1stage` 的步骤逐步实现**：
   - 求和结果先 `downcast` 成 T 再 `upcast` 回来；
   - 加上 residual（fp32）；
   - residual_out 写舍入后的值；
   - sumsq 用 fp32 未舍入的值，`square_sum += v*v`；
   - `out = acc * rms * w`，gemma 模式下 w+1。
4. **block 按行对齐**：pack→block 的映射还是 `pack / BS`。因为 `(H/8) % BS == 0`，一个 block 不会跨行。
   - **一行一个 block（row，BS = H/8）**：直接调用 1-stage 用的 `ar_fusion_epilogue_block_reduce<AddFunctor, float, 32>`。
   - **一行多个 block（bs64/bs128）**：
     - 每 32 lane 用 `warpReduce<AddFunctor,float,32>` 算出 partial；
     - nt store 到 rank 本地的 `partials[round][pack/32]`。写之前 canonicalize：0xFFFFFFFF 改成 0x7FFFFFFF。同时 rearm `partials[rearm][...]`。
     - 前 32 个 lane 各自轮询这一行的一个 partial，超出行宽的 lane 取 0；
     - 再用 1-stage 跨 warp 的那个 butterfly（`warpReduceRuntime`，宽度取 ≥ 组数的 2 的幂）折叠；
     - thread 0 计算 `rsqrtf(sum/hidden + eps)`。

     这样三个变体的 out 都和 1-stage 逐位一致。
5. **partial slot 的位置**：紧接在本 rank 的 inbox 后面，同一块 uncached buffer，init 时一起 memset 成 0xFF。大小是 `3 × kOneShotRmsMaxGroups × 4B`，其中 `kOneShotRmsMaxGroups = kOneShotMaxPacks/32`。
6. **为什么 state 要固定 hidden 和 BS**：partial 交换要求同一行的几个 block 用同一个 round。只要 hidden 和 BS 不变，同一行的 block 永远一起启动（或者都不启动），各自的计数器自然保持同步。反过来，如果同一个 inbox 被换了 H 去用，行和 block 的对应关系就变了，会死锁。Gluon kernel 也有同样的限制。
7. **ISA 检查**（bf16、tp8、bs64，独立编译）：
   - VGPR 72，ScratchSize 0，LDS 136B；
   - 7 条远端 nt store 之后才是 residual/weight 的 load，然后进入轮询；
   - 两个轮询循环里都是 nt load，没有被提到循环外。

   检查方法：参考 `LAMPORT_ONESHOT_PROGRESS.md` 第 3 节的 hipcc 命令，实例化 `lamport_allreduce_rmsnorm_oneshot<opus::bf16_t, 8, 64>`。

### 4.4 测试（`test_lamport_oneshot_rmsnorm.py`）

候选：

| 名字 | 是什么 |
|---|---|
| `car_fused` | 生产路径 `tensor_model_parallel_fused_allreduce_rmsnorm`：小尺寸走 1-stage，超过 `128*7168*2/tp` 字节走 split |
| `car_1stage` | `ca.custom_fused_ar_rms(..., use_1stage=True)`，强制走现有 1-stage（pull 加两次 barrier） |
| `lamport_bs64` / `lamport_bs128` | 切行版本，每行 H/8/bs 个 block |
| `lamport_row` | 一行一个 block（bs = H/8：7168 对应 896，8192 对应 1024） |
| `lamport_ar+norm` | 纯 Lamport AR（bs64）之后跑 `rmsnorm2d_fwd_with_add`：同样的协议，不融合 |
| `lamport_ar_only` | 只跑 AR，不算候选，用来拆分开销 |

- **计时**：一个 CUDA graph 里放 10 次调用，回放 20 次取中位数，报告最慢那张卡。每列都给出 us、TB/s（每个 rank 的消息字节数除以时间）和 err。
- **正确性检查**：
  - Lamport 融合版的 out 和 residual_out 与 `car_1stage` 逐位比较，记在 `bitdiff` 列（NaN 和 NaN 算相等）；
  - 所有候选都和 torch 参考比较。参考是 fp32 按 rank 顺序求和、舍入后加 residual、再做 fp32 RMSNorm；gfx942 上 bf16 用截断，gfx950 上是 RNE（opus 的 `OPUS_FP32_to_BF16_DEFAULT`）。
  - `check_size_changes`（eager）：M 按 cap→1→cap→3→1→cap/2+1→cap 变化，重复 3 遍，每隔 3 个注入一次 0xFFFF，和 `car_1stage` 逐位比较，记在 `resize bitdiff` 列。
- **输出**：一张完整的 summary 表，加一张对比表（每个候选的 us、最优 Lamport 变体、相对各基线的 speedup）。

---

## 5. 测试结果（MI308X，bf16，3 次全量取每格中位数，单位 µs）

### 5.1 结果表

```
| tp | (m, n)     | KB  | car_fused | car_1stage | lamport_bs64 | lamport_bs128 | lamport_row | lamport_ar+norm | lamport_ar_only | best  | vs car_fused |
|  2 | (1, 7168)  |  14 |     10.04 |      10.03 |         9.34 |          8.90 |        8.19 |            9.94 |            5.82 | row   |         1.23 |
|  2 | (2, 7168)  |  28 |      9.95 |       9.92 |        10.15 |          8.96 |        8.39 |           10.04 |            5.91 | row   |         1.19 |
|  2 | (4, 7168)  |  56 |     10.39 |      10.38 |        10.47 |          9.87 |        8.49 |           10.80 |            6.69 | row   |         1.22 |
|  2 | (8, 7168)  | 112 |     11.59 |      11.60 |        12.30 |         11.36 |        9.80 |           12.12 |            8.08 | row   |         1.18 |
|  2 | (16, 7168) | 224 |     14.11 |      14.07 |        15.06 |         14.48 |       12.36 |           14.69 |           10.56 | row   |         1.14 |
|  2 | (32, 7168) | 448 |     19.22 |      18.08 |        20.37 |         19.75 |       17.44 |           19.10 |           15.37 | row   |         1.10 |
|  2 | (1, 8192)  |  16 |     10.22 |      10.22 |         9.84 |          8.86 |        8.10 |            9.94 |            5.66 | row   |         1.26 |
|  2 | (2, 8192)  |  32 |     10.19 |      10.13 |         9.74 |          9.11 |        8.34 |           10.13 |            5.98 | row   |         1.22 |
|  2 | (4, 8192)  |  64 |     10.80 |      10.81 |        10.54 |         10.11 |        8.91 |           11.08 |            6.44 | row   |         1.21 |
|  2 | (8, 8192)  | 128 |     12.11 |      12.14 |        12.44 |         11.53 |       10.11 |           12.43 |            8.24 | row   |         1.20 |
|  2 | (16, 8192) | 256 |     14.95 |      14.97 |        15.32 |         14.60 |       13.19 |           15.54 |           11.09 | row   |         1.13 |
|  2 | (32, 8192) | 512 |     20.74 |      20.65 |        21.68 |         20.76 |       18.98 |           21.07 |           16.80 | row   |         1.09 |
|  4 | (1, 7168)  |  14 |     10.73 |      10.76 |         9.68 |          9.19 |       10.91 |           10.69 |            6.48 | bs128 |         1.17 |
|  4 | (2, 7168)  |  28 |     10.70 |      10.73 |        10.69 |          9.62 |       10.44 |           11.13 |            6.96 | bs128 |         1.11 |
|  4 | (4, 7168)  |  56 |     11.12 |      11.10 |        11.58 |         11.12 |       11.22 |           12.25 |            7.38 | bs128 |         1.00 |
|  4 | (8, 7168)  | 112 |     12.45 |      12.49 |        13.08 |         12.60 |       12.20 |           14.95 |           10.71 | row   |         1.02 |
|  4 | (16, 7168) | 224 |     16.33 |      16.34 |        17.38 |         17.07 |       15.99 |           17.67 |           13.44 | row   |         1.02 |
|  4 | (32, 7168) | 448 |     22.75 |      22.44 |        24.99 |         24.20 |       23.66 |           24.34 |           20.28 | row   |         0.96 |
|  4 | (1, 8192)  |  16 |     11.04 |      11.01 |        10.08 |          9.43 |       10.66 |           10.60 |            6.30 | bs128 |         1.17 |
|  4 | (2, 8192)  |  32 |     11.08 |      11.08 |        10.34 |         10.30 |       10.87 |           10.86 |            6.84 | bs128 |         1.08 |
|  4 | (4, 8192)  |  64 |     11.44 |      11.47 |        11.85 |         11.07 |       11.28 |           12.44 |            8.33 | bs128 |         1.03 |
|  4 | (8, 8192)  | 128 |     13.00 |      13.00 |        14.57 |         12.36 |       12.64 |           15.01 |           11.61 | bs128 |         1.05 |
|  4 | (16, 8192) | 256 |     17.23 |      17.26 |        18.70 |         17.08 |       16.56 |           18.85 |           14.50 | row   |         1.04 |
|  4 | (32, 8192) | 512 |     22.69 |      23.60 |        27.79 |         26.64 |       25.06 |           25.50 |           20.93 | row   |         0.91 |
|  8 | (1, 7168)  |  14 |     11.18 |      12.49 |        10.86 |         10.26 |       14.50 |           11.68 |            7.30 | bs128 |         1.09 |
|  8 | (2, 7168)  |  28 |     12.76 |      12.76 |        11.29 |         10.54 |       14.49 |           11.88 |            7.78 | bs128 |         1.21 |
|  8 | (4, 7168)  |  56 |     13.14 |      13.20 |        12.08 |         12.17 |       14.90 |           13.41 |            9.68 | bs64  |         1.09 |
|  8 | (8, 7168)  | 112 |     14.46 |      14.35 |        15.95 |         14.71 |       16.25 |           17.52 |           13.08 | bs128 |         0.98 |
|  8 | (16, 7168) | 224 |     18.22 |      16.31 |        20.15 |         19.71 |       19.24 |           21.76 |           17.37 | row   |         0.95 |
|  8 | (32, 7168) | 448 |     20.38 |      24.67 |        32.32 |         32.12 |       26.26 |           30.75 |           26.67 | row   |         0.78 |
|  8 | (1, 8192)  |  16 |     12.84 |      12.80 |        10.40 |         10.25 |       14.95 |           11.52 |            7.65 | bs128 |         1.25 |
|  8 | (2, 8192)  |  32 |     11.64 |      13.08 |        11.26 |         10.92 |       15.18 |           12.16 |            7.97 | bs128 |         1.07 |
|  8 | (4, 8192)  |  64 |     13.58 |      12.10 |        12.31 |         12.12 |       15.52 |           13.88 |            9.79 | bs128 |         1.12 |
|  8 | (8, 8192)  | 128 |     14.57 |      13.39 |        16.28 |         15.38 |       17.02 |           18.02 |           13.67 | bs128 |         0.95 |
|  8 | (16, 8192) | 256 |     16.01 |      17.44 |        21.31 |         21.54 |       20.31 |           22.95 |           18.85 | row   |         0.79 |
|  8 | (32, 8192) | 512 |     20.20 |      26.32 |        35.11 |         36.63 |       29.16 |           33.08 |           28.55 | row   |         0.69 |
```

- 原始日志在 `/tmp/lamport_rms_full.log`、`/tmp/lamport_rms_full_run2.log`、`/tmp/lamport_rms_full_run3.log`（/tmp 里的东西可能会被清掉）。
- 第一遍有两个约 2 倍的离群点（TP8 1×7168 的 bs128 是 20.9，TP4 1×7168 的 row 是 21.1），所以多跑了两遍，取中位数。

### 5.2 结论

- **对比现有 fused（car_fused）**：
  - TP2：全部尺寸都赢，1.09–1.26x。
  - TP4：64KB 以内 1.00–1.17x，112–256KB 持平（1.02–1.05），448–512KB 输（0.91–0.96）。
  - TP8：64KB 以内 1.07–1.25x，112KB 起输（0.69–0.98）。

  大尺寸输是因为 one-shot 要推 (N−1) 份完整数据，和纯 Lamport AR 的结论一致。car_fused 在 TP8 下 M 稍大时会切到 split（H=7168 时 M=32、H=8192 时 M≥16）。
- **切不切行和 3.3 节的讨论一致**：
  - TP2：row 最快，每行活少，切了只会多一次交换。
  - TP4/TP8 小 M：必须切。比如 TP8 1×8192 下 row 是 14.95 µs，bs128 是 10.25 µs。
  - M ≥ 16 时 row 又追上来，因为行数本身已经能把 GPU 铺满。
- **融合 vs Lamport AR + 单独 norm**：所有 case 融合版都更快，1.02–1.27x。
- **收益没有预期大，epilogue 是瓶颈**：
  - 只看协议，TP8 1×7168 下纯 AR 是 7.30 µs，现有 1-stage 是 12.49 µs，差约 5 µs。融合后是 10.26 µs，epilogue 吃掉了约 3 µs。
  - 小 M 时，融合版普遍比纯 AR 慢 2–3.5 µs。其中跨 block 交换 partial 大约占 0.7 µs（TP2 下 bs128 对比 row）。
  - row 版本完全没有交换，也比纯 AR 慢约 2.4 µs（TP2 1×7168：8.19 对比 5.82）。这部分可能来自 epilogue 的额外读写和 3 次 `__syncthreads`，也可能是整行压在一个 CU 上导致的，还没拆清楚。

### 5.3 正确性

- **性能用例**：3 TP × 2 H × 6 M × 3 遍，所有 Lamport 融合版的 out 和 residual_out 都和 `car_1stage` 逐位一致，torch 参考的 err 都是 0。
- **resize 检查里的唯一差异**：TP4、H=7168、M=45（case i=20，没有注入 NaN）时，`out[29, 1150]` 一个元素对不上，3 遍都能稳定复现。
  - Lamport 三个变体都是 0x3eb6，torch 参考也是 0x3eb6；现有 `car_1stage` 是 0x3eb5，差 1 个 bf16 ulp。
  - residual_out 是逐位相同的，所以差别出在 norm 那一步的浮点末位。
  - 具体是哪条指令导致的没有继续查。编译选项是 `-O3 -fgpu-flush-denormals-to-zero`，没有 fast-math。
  - 复现脚本：`/tmp/lamport_rms_debug/debug_resize.py`，见附录 A。

---

## 6. 待办 / 下次继续

1. **[优先] 拆分 epilogue 那 2–3 µs 的开销**：用 rocprofv3 的 kernel trace 或 thread trace，对比 `lamport_row` 和纯 AR（bs64）、bs128 和 row。
   - 要分清的几个来源：额外的 residual/weight load 和 out/residual_out store；3 次 `__syncthreads`；整行压在一个 CU 上；partial 交换。
   - 目标：epilogue 压到 0.5 µs 左右的话，小尺寸能到 1.3–1.4x。
2. **partial 交换做 XCD 本地化**：MI308X 是 4 个 XCD，block 按 ID 轮流分配。可以把 blockIdx 重映射，让同一行的 block 落在同一个 XCD，通过 L2 交换。
   - 注意 1：gfx942 上 agent scope 的一致性要用 sc1，开销和 uncached 差不多；只依赖同一个 XCD 的 L2，就要假设分派方式。分派假设一旦不成立，可能直接死锁，必须有兜底。
   - 注意 2：gfx9 的 vmcnt 同时计 load 和 store，并且按序完成，所以轮询前会先等前面所有 store 被 ack。可以考虑用标量 load（带 glc，走 lgkmcnt）来轮询 partial，绕开这个等待。
3. **调度门槛**（初步，最好再细扫一遍）：
   - TP2：≤512KB 用 row；
   - TP4：≤128KB 用 bs128（小 M）或 row；
   - TP8：≤64KB 用 bs128。
   - 超过这些门槛继续走现有的 fused 路径。
4. **接入 `fused_allreduce_rmsnorm` 的调度**：Python 端在 `_init_ipc` 里创建 buffer、`init_oneshot_rmsnorm`，然后 barrier。每个 hidden 一个 state，一个模型通常只有一个 hidden。
5. **扩展**：per-token FP8 epilogue（注意和 #7082 的 `round_before_quant` 语义对齐）、gemma_norm 的测试、fp16 的测试。
6. **gfx1250 移植**（结合 `LAMPORT_ONESHOT_DESIGN.md`）：
   - wave32，block 最小 32；
   - 访存要换成 system scope 的 b128 原语；
   - partial 交换可以用 cluster（最多 16 个 WG、同时启动、有 barrier，但不能共享 LDS）；
   - 测试参考的 bf16 舍入改成 RNE。
7. **（可选）** 查清 TP4 那 1 个元素 1 ulp 差异的来源：对比两个 kernel 的 ISA。
8. **（可选）** 本地 main 落后 263 个 commit，pull 之后要重新测一遍：
   - #6954 统一了求和顺序；
   - #6640 给 `module_custom_all_reduce` 加了 `-DOPUS_FP32_to_BF16_DEFAULT=5`（软件 RNE，可以用同名环境变量覆盖）。pull 之后 gfx942 上 bf16 也是 RNE，测试里 `round_like_kernel` 的截断分支要跟着改（纯 Lamport 的 `test_lamport_oneshot.py` 也一样）。

   这两点都会影响逐位比较和基线。

---

## 附录 A：调试脚本（原本放在 /tmp/lamport_rms_debug/，这里留一份）

`debug_resize.py`：在指定 TP 下重跑 resize 检查，打印所有和 `car_1stage` 不一致的元素。
```bash
PYTHONPATH=$PWD:$PWD/op_tests/multigpu_tests python3 /tmp/lamport_rms_debug/debug_resize.py <tp> <reps> [n ...]
```
```python
import sys
from multiprocessing import Pool, set_start_method
import torch
import test_lamport_oneshot_rmsnorm as t
from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel,
    ensure_model_parallel_initialized, get_tp_group, init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port


def worker(tp, rank, ns, reps, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method=init_method)
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    dtype = dtypes.bf16
    states = t.init_states(ca, ns)
    found = []
    for n in ns:
        cap = t.ONESHOT_MAX_BYTES // (n * dtype.itemsize)
        ms = [cap, 1, cap, 3, 1, cap // 2 + 1, cap]
        for name in t.LAMPORT_FUSED:
            for i, m in enumerate(ms * reps):
                nan = i % 3 == 0
                x, xs, res, w = t.make_inputs(tp, rank, 1000 + i, m, n, dtype, nan=nan)
                out, res_out = t.lamport_fused(states[(n, name)], x, res, w)
                b_out, b_res = t.car_1stage(ca, x, res, w)
                r_out, r_res, _ = t.run_torch(xs, res, w, t.EPS)
                for what, a, b, r in (("out", out, b_out, r_out), ("res", res_out, b_res, r_res)):
                    bad = ~((torch.isnan(a) & torch.isnan(b)) | (a.view(torch.int16) == b.view(torch.int16)))
                    if bad.any():
                        for (ri, ci) in bad.nonzero().tolist()[:4]:
                            found.append(dict(rank=rank, n=n, var=name, i=i, m=m, nan=nan, what=what,
                                              row=ri, col=ci,
                                              lamport=hex(a[ri, ci].view(torch.int16).item() & 0xffff),
                                              car=hex(b[ri, ci].view(torch.int16).item() & 0xffff),
                                              ref=hex(r[ri, ci].view(torch.int16).item() & 0xffff),
                                              nbad=int(bad.sum())))
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return found


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    tp, reps = int(sys.argv[1]), int(sys.argv[2])
    ns = [int(v) for v in sys.argv[3:]] or [7168, 8192]
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [pool.apply_async(worker, (tp, r, ns, reps, init_method)) for r in range(tp)]
        res = [j.get(timeout=900) for j in jobs]
    allf = [f for r in res for f in r]
    print(f"TOTAL mismatching (case, tensor) records: {len(allf)}")
    for f in allf[:40]:
        print(f)
```

`median_tables.py`：对多次运行的 "vs existing fused" 表逐格取中位数。
```bash
python3 /tmp/lamport_rms_debug/median_tables.py run1.log run2.log run3.log
```
```python
import statistics, sys
import pandas as pd

LAM = ["lamport_bs64", "lamport_bs128", "lamport_row"]


def parse(path):
    txt = open(path).read()
    blk = txt.split("vs existing fused (markdown):\n")[1]
    lines = [l for l in blk.splitlines() if l.startswith("|")]
    hdr = [h.strip() for h in lines[0].strip("|").split("|")]
    return [dict(zip(hdr, [c.strip() for c in l.strip("|").split("|")])) for l in lines[2:]]


runs = [parse(p) for p in sys.argv[1:]]
rows = []
for cells in zip(*runs):
    r = {"tp": cells[0]["tp"], "(m, n)": cells[0]["(m, n)"], "KB": float(cells[0]["KB"])}
    for k in cells[0]:
        if k.endswith(" us"):
            r[k] = statistics.median(float(c[k]) for c in cells)
    best = min(LAM, key=lambda k: r[f"{k} us"])
    r["best lamport"] = best
    r["vs car_fused"] = r["car_fused us"] / r[f"{best} us"]
    r["vs car_1stage"] = r["car_1stage us"] / r[f"{best} us"]
    r["vs lamport_ar+norm"] = r["lamport_ar+norm us"] / r[f"{best} us"]
    r["fused - ar_only us"] = r[f"{best} us"] - r["lamport_ar_only us"]
    r["max err"] = max(float(c["max err"]) for c in cells)
    r["max bitdiff"] = max(float(c["max bitdiff"]) for c in cells)
    rows.append(r)
print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))
```
