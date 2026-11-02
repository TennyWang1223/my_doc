# Lamport AR + RMSNorm：切分 H 与性能（讨论记录）

> - **整理日期**：2026-10-30；2026-11-02 补充（实测时钟、TA 模型的微基准验证、uncached/nt 的缓存行为）
> - **来源**：2026-10-30 的会话。问题是"`lamport_allreduce_rmsnorm_oneshot` 里切分 H 和性能是什么关系"
> - **微基准**：`/mnt/raid0/tiwang/work/demo/ta_vmem/`（`ta_split.cpp`、`copy_bs.cpp`、`uc_latency.cpp`、`l1_policy.cpp`、`clk.cpp`、`cus.cpp`），用 `hipcc -O3 --offload-arch=gfx942` 编译，结果见 §1.5、§3.4、§3.5
> - **代码**：`csrc/include/custom_all_reduce.cuh`，kernel `lamport_allreduce_rmsnorm_oneshot`（约第 2466 行），launch 在 `OneShotRmsState::run`（约第 2675 行）
> - **相关文档**：
>   - `fused_poc_notes.md`（同目录，下文简称 NOTES）：融合 POC 记录，实测数据在 §5，待办在 §6
>   - `LAMPORT_ONESHOT_PROGRESS.md` §5.2：纯 Lamport AR 的 bs64 / bs128 / bs512 扫描
>   - MI300 带宽白皮书：`/home/tennwang/gpu-docs/mi300/perf/mi300_gpu_bandwidth.pdf`（表里包含 MI308）
> - **附录**：同一次会话里画的 custom allreduce 1-shot / 2-shot 数据流图（TP4），以及顺带发现的两个问题

## 0. 下次接着看这里

**结论**：切分 H 不改变总的读写量，只改变两件事：
1. **一行的读写由几个 CU 来发。** 这是收益：单个 CU 的读写部件有硬上限（§1.4），不切时一行全压在一个 CU 上。
2. **整行的 sum of squares 在哪里汇总。** 这是代价：不切时在 block 内汇总，几乎免费；切了只能跨 block 走显存（§4）。

TP 越高，一行的活越多，越需要多个 CU 分摊；M 越大，不切也已经有很多 CU 在干活。所以只有"高 TP、小 M"时才值得切（§5）。

**更正过的数**：讨论中途先按 2 GHz、后按 1.85 GHz（`rocm-smi --showclkfrq` 的最高档）换算过，都偏乐观。实测负载下 SCLK 只有 **1.42 GHz**（`clk.cpp`：1～320 个 block 跑 5 µs～3 s，`clock64()` 都是 1.41～1.44 GHz；同时 `rocm-smi --showclocks` 显示 `sclk 1418Mhz`）。本文的时间现在都按 1.42 GHz 算：一条 dwordx4 占 TA 16 cycle ≈ **11.3 ns**。

**补充的结论（2026-11-02）**：
- 单 CU 的 TA 模型用微基准验证过：一行压在一个 CU 上时，每多一条 dwordx4 指令，kernel 慢 0.16 µs，和 14 wave × 11.3 ns 吻合（§3.4）。
- 把一行分给更多 block 有一个固定代价，实测约 0.45 µs，和跨 block 同步无关。所以切分只在"最忙的 CU 要发 40 条以上 dwordx4"时才划算（§3.4）。纯搬运 kernel 用 block_size=64 不会比 128/256 更快（§3.5）。
- `hipDeviceMallocUncached` 对**本地**读几乎没有影响：普通 load 照样命中 L1 和 L2。它只让**跨卡**访问不进读取方的 L2。轮询不会读到 L1 旧值，靠的是 nt 或 sc 位（§1.5）。

**还没弄清的**：
- 切分版的跨 block 汇总在 M 大时代价被放大到好几 µs，具体是哪一环被放大还没拆开（§4.3）。
- §3 的"单 CU 瓶颈"在微基准上已经验证，但在真实 kernel 里还没用计数器确认（§8）。
- 上面那 0.45 µs 的固定代价具体来自哪里，还没拆开（§3.4）。

---

## 1. 硬件基础：一条读写指令走的路

### 1.1 基本单位

- **CU**：GPU 里的一个计算单元。MI308X 一张卡有 80 个：4 个 XCD × 4 个 SE × 5 个 CU（`cus.cpp` 读 `HW_ID`/`XCC_ID` 寄存器实测）。一个 block 只会放在一个 CU 上运行。
- **XCD / SE**：XCD 是一块计算 die，有自己的 L2；SE（Shader Engine）是 XCD 内的一组 CU。MI300（gfx942）**没有** GL1 / L1.5 这一层（GL1 是 RDNA 和 MI400 才有），SE 内的 CU 之间只共享 SQC（指令缓存和标量缓存，两个 CU 共用一个），不共享向量数据缓存。所以对向量读写来说，要关心的边界是 XCD（同一个 L2），不是 SE。
- **block 怎么分到 CU**：按 block ID 轮流分到 4 个 XCD（`blockIdx % 4` 相同的 block 在同一个 XCD）。实测 M=1、bs64 的 14 个 block 落在 14 个不同的 CU 上。
- **wave**：64 个线程绑在一起，同一时刻执行同一条指令。每个线程叫一个 **lane**。
- **VGPR**：每个线程自己的寄存器。
- **VMEM 指令**：每个 lane 各自去显存读或写的指令，就是 kernel 里的 global_load / global_store。访问 CU 内共享内存（LDS）的指令走另一条路。
- **cycle（周期）**：GPU 计算核心时钟（SCLK）的一拍。按实测的 1.42 GHz，一拍约 0.70 ns（最高档 1.85 GHz 在负载下跑不到，见 §0）。

### 1.2 路线图

```
 one CU  (MI308X has 80; a block always runs inside one CU)
+------------------------------------------------------------------+
|  SIMD0       SIMD1       SIMD2       SIMD3    <- run the waves   |
|  [VGPRs]     [VGPRs]     [VGPRs]     [VGPRs]  <- per-thread regs |
|     |           |           |           |                        |
|     +-----------+-----+-----+-----------+                        |
|                       |  data bus: 64 bytes per cycle, shared by |
|                       v  all waves on this CU, loads + stores    |
|   +----------------------------------------------------------+   |
|   | TA   address unit: receives each load/store instruction, |   |
|   |      hands its 64 lane addresses to L1, 4 per cycle      |   |
|   | TCP  this CU's L1 data cache (32 KB)                     |   |
|   |      LFIFO: table of loads whose data has not come back  |   |
|   | TD   data unit: writes returned data into the VGPRs      |   |
|   +----------------------------------------------------------+   |
+-------------------------------|----------------------------------+
                                v
              L2 cache: 4 MB, shared by the 20 CUs of one XCD
              (no GL1 / L1.5 level on MI300)
                                v
              data fabric -> this GPU's HBM      -> XGMI -> other GPUs
                                                  (the push goes here)
```

一个 CU 只有一套这样的部件（白皮书里叫 VMEM pipeline），这个 CU 上 4 个 SIMD 的所有 wave 的读写都要排队从这里经过（白皮书 §3.2："A single VMEM pipeline services memory requests from active waves on all four SIMD units"）。

路径是固定的：TA → TCP（L1）→ TCC（本 XCD 的 L2）→ data fabric。不管内存是不是 uncached、指令带不带 nt，都走这条路，只是在 L1、L2 上"能不能命中、留不留下"不同（§1.5）。

### 1.3 名词解释

- **VMEM 管线**：图里 SIMD 下面那条数据线，加上 TA、TCP、TD 三个部件。一头接寄存器，一头接 L1，L1 再往外接 L2 和显存。
- **L1D / vL1D**：每个 CU 自己的一级数据缓存，32 KB。
- **TCP**：就是 L1 这个部件在 AMD 文档里的名字（Texture Cache per Pipe，"Texture"是历史叫法，和纹理已经无关）。L2 在文档里叫 TCC，每个 XCD 一个，4 MB，被 20 个 CU 共用。
- **TA**：L1 前面的地址处理部件（Texture Addresser）。wave 发出的读写指令先到 TA，TA 把 64 个 lane 的地址整理好，每周期交 4 个给 L1 去查。
- **TD**：把读回来的数据写进寄存器。
- **64 B/cycle**：寄存器和 L1 之间那条数据线的宽度。传的是数据本身：load 时是读回来的数据，store 时是写出去的数据。每周期最多 64 字节，这个 CU 上所有 wave 的读和写共用。
- **coalesce（合并）**：相邻几个 lane 访问的地址正好挨着时，硬件把它们当成一个地址处理。MI300 只有"每 lane 读写 4 字节"的指令能合并，最多 4 个 lane 合成 1 个；每 lane 16 字节的指令不能合并，64 个地址只能一个个处理。
- **dword / dwordx4**：dword 是 4 字节；dwordx4 是每个 lane 一次读写 16 字节。
- **1 KB**：一条 dwordx4 指令一共搬的数据量，64 lane × 16 B。不是指令本身的大小。在本 kernel 里就是一个 wave 的 64 个线程各搬一个 pack（8 个 bf16）。
- **16 cycle**：一条 dwordx4 指令占用 TA 出口的时间（64 个地址 ÷ 每周期 4 个）。这段时间出口被它占满，后面的指令在 TA 的缓冲里排队。它是吞吐上的代价，既不是 wave 发指令花的时间，也不是数据在路上的时间。1 KB ÷ 16 cycle = 64 B/cycle，数值上恰好等于数据线宽度，但这是两个独立的上限（见 §1.4）。
- **延迟**：请求发出到数据回来的时间，比 16 cycle 长得多。这段时间 TA 不被占用，可以继续处理后面的指令。
- **LFIFO**：L1 里的一张"未完成 load 登记表"。load 请求发出后、数据回来前，要在表里占位置，记住数据回来后送到哪个 wave 的哪些寄存器；数据回来才腾出位置。表满了，新的 load 就发不出去。
- **704 项 / 16 项**：登记表一共 704 个格子。一个格子记 4 个 lane 的数据（64 字节），一条 dwordx4 load 有 64 个 lane，所以占 16 个格子。
- **在途**：请求已经发出、数据还没回来。

### 1.4 单个 CU 的两道关

- **收费口（吞吐）**：一个 CU 每 16 个周期最多放行一条 dwordx4 指令（1 KB），读和写都算。换算成带宽：64 字节/周期 × 1.42 GHz ≈ 91 GB/s（按最高档 1.85 GHz 是 118 GB/s），这是一个 CU 上所有 wave 加起来的上限。整卡 80 个 CU 合起来比显存带宽还大，平时感觉不到；但如果一行的所有读写只能走一个 CU，就会卡在这个数上。细节：
  - **16 cycle 怎么来的**：TA 出口每周期交 4 个地址给 L1。一条指令要交几个地址看能不能合并：dwordx4（以及 dwordx2）不能合并，64 个 lane 就是 64 个地址，要 16 cycle；dword 如果相邻 4 个 lane 地址连续，4 个合成 1 个，只剩 16 个地址，4 cycle 交完；不连续的 dword 也要 16 cycle。本 kernel 的读写全是 dwordx4，所以都按 16 cycle 算。
  - **1 KB 怎么来的**：一条 dwordx4 指令 64 lane × 每 lane 16 B = 1024 B，是这条指令搬的数据量（白皮书 Table 5 的 per-instruction 一列）。1 KB ÷ 16 cycle = 64 B/cycle。
  - **TA 不是"做完一条才收下一条"**：TA 入口每 4 cycle 能收一条指令，先放进内部缓冲（FIFO）排队；出口只有一个，每周期 4 个地址。一条指令的地址交完它就离开 TA，不等数据回来，所以很多条可以同时在 L1/L2/显存上在途（上限见下一条）。限制的只是出口速率：不管缓冲里排了几条、来自几个 wave，合起来每周期只能交 4 个地址。dwordx4 的入口速率（4 cycle 一条）比出口（16 cycle 一条）快 4 倍，连续发十来条缓冲就满了，CU 被反压，之后只能 16 cycle 发一条（白皮书 §3.3：连续 12 条就会撑满）。
  - **和数据线是两个上限**：寄存器和 L1 之间的数据线也是 64 B/cycle。dwordx4 和合并良好的 dword（256 B / 4 cycle）在 TA 出口上也是 64 B/cycle，两者恰好相等；但 dwordx2 是 512 B / 16 cycle = 32 B/cycle，不连续的 dword 是 256 B / 16 cycle = 16 B/cycle，会先卡在 TA，用不满数据线。
- **路上（在途）**：704 ÷ 16 = 44，一个 CU 上最多 44 条 dwordx4 load 同时在途，共 44 KB。每条 load 都要等一个完整往返，所以每过一个往返时间，这个 CU 最多拿回 44 KB。往返越慢（读显存、跨卡），读得越慢，和数据线宽不宽无关。
- **比喻**：TA 是只有一个收费口的收费站，收费口前有一段排队车道；16 cycle 是每辆车在收费口停留的时间；车开到目的地再开回来的时间是延迟，和收费口无关；路上同时最多跑 44 辆车。
- **wave 是怎么被卡住的**：wave 把指令交给 TA 只要几个周期，交完继续往下执行，直到真要用读回来的数据才停下来等。但如果 TA 前面已经排满，wave 连指令都交不出去，只能干等。

出处（白皮书）：§2.2、§3.2（64 B/cycle）；§3.3（TA；dwordx4 不能合并，一条占 16 cycle）；§3.4.7 和 Table 8（LFIFO 704 项，每条 load 占 16 项）；§3.1.1（block 按轮转分到各个 XCD）。

### 1.5 uncached 内存、nt 和 sc 位实际做了什么

Lamport 的 inbox 和 partial 都在 `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` 申请的内存上（`csrc/kernels/custom_all_reduce.cu` 约第 546 行），读写用 `__builtin_nontemporal_load/store`，编译出来是带 `nt` 的 `global_load/store_dwordx4`。

**配置链路**（读源码确认）：
- `hipDeviceMallocUncached` → ROCr 的 Uncached 标志 → KFD 的 `ALLOC_MEM_FLAGS_UNCACHED` → `AMDGPU_GEM_CREATE_UNCACHED` → 驱动在 GC 9.4.3 上把页表项的 **MTYPE 设成 UC**（`/usr/src/amdgpu-6.14.14-*/amd/amdgpu/gmc_v9_0.c` 约第 1205 行）。
- peer 通过 IPC 导入时，UNCACHED 标志会被带过去（`amdgpu_dma_buf.c` 约第 690 行），所以 peer 映射这块内存时也是 UC。
- MTYPE 是**每张卡自己页表里的属性**。"绕过 cache"指的是发起访问那张卡的 cache。

**名词**：
- **nt**：指令上的 non-temporal 位。ISA 只写了 "data is streamed or uncached"，没有规定 L1 的具体行为。
- **sc1 / sc0 sc1**：指令上的作用域位，分别表示 device（agent）和 system 作用域。`__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)` 编译成 `sc1`，SYSTEM 作用域编译成 `sc0 sc1`。这是内存模型保证一致性的正式手段。
- **过期（stale）**：别人已经改了内存，cache 里还是旧副本。这和"脏"（自己写了还没写回）不是一回事。

**实测 1：各级 cache 会不会命中**（`uc_latency.cpp`，单 lane 依赖链读，单位 ns；L1 命中约 85，L2 命中约 150，本地显存或 Infinity Cache 约 275，跨卡约 500）：

| buffer | 普通 load：8 行（装得进 L1） | 普通 load：256 KB（装得进 L2） | nt load：8 行 | nt load：256 KB |
|---|---|---|---|---|
| 本地 hipMalloc | 90 | 156 | 162 | 148 |
| 本地 UC | **87** | **148** | **146** | 151 |
| peer hipMalloc（MTYPE NC） | 88 | **318** | 144 | 375 |
| peer UC | **88** | **491** | 588 | 604 |

**实测 2：nt / sc 会不会命中 L1，会不会把数据留在 L1**（`l1_policy.cpp`：先用 A 方式把 128 行读 4 遍，再用 B 方式只读 1 遍并计时。只读一遍，B 自己没机会把数据装进 L1，所以测出来的是 A 留下了什么）：

| 先读方式 → 再读方式 | 本地 hipMalloc | 本地 UC | peer UC |
|---|---|---|---|
| 普通 → 普通 | **84** | **87** | **87** |
| 普通 → nt / sc1 / sc0 sc1 | 147～150 | 145～150 | 587～591 |
| nt → 普通 | 145 | 154 | 722 |
| sc1 → 普通 | 145 | 151 | 543 |

注意：地址在 wave 内统一的普通 load 会被编译器改成 `s_load_dword`，走标量缓存，不走 L1。测试里用 inline asm 造了一个 VGPR 偏移，强制编译成向量 load，并在汇编里核对过。

**结论**：
- **L1**：UC 不影响 L1。普通 load 读 UC 内存（本地或 peer）照样命中 L1。nt、sc1、sc0 sc1 的 load 即使数据就在 L1 里也不命中，读完也不把数据留在 L1。
- **本地 UC 的 L2**：读照样命中本地 L2（148 ns，和 hipMalloc 一样）。读这块内存时，UC 基本不起作用。按 MI300A MAS（"Coherency improvements"一节），UC 的**写**不留在 L2（这点没实测）。
- **peer UC 的 L2**：UC 绕过的是**读取方自己的** L2（256 KB 跨度上 491 ns，NC 只要 318 ns）。src 卡的 L2 本来就不在跨卡访问的路径上。
- **本地 L2 为什么不会给出过期数据**：本地内存的写者有两类，peer 经 XGMI 写进来，或者本卡另一个 XCD 写。两者都要经过 data fabric。MI300 在 data fabric 里用 shadow tag 记录哪些 L2 读过某一行，有人写这一行时发探测把那份副本失效（同一节 MAS）。反过来也能印证：如果 L2 的旧副本不被失效，Lamport 用 nt 轮询会永远等不到数据。
- **对 Lamport 的意义**：
  - 在同一个 kernel 里轮询，必须用 nt 或 sc 位；用普通 load 可能会一直读 L1 里的旧值（不同 kernel 之间没问题，kernel 开始时 L1 会被失效）。sc 位是内存模型明文保证的，nt 不命中 L1 只是实测行为，所以改用 `__hip_atomic_load(relaxed, agent/system)` 更稳妥。
  - 本地 inbox 没有变化时，一轮 nt 轮询大约是一次 L2 命中（约 150 ns）；peer 写入之后的那一轮要回显存（约 275～385 ns）。
  - 不管哪种缓存属性，指令都照样占 TA 的 16 cycle 和 LFIFO 的位置。
- **还没解释的现象**：读 peer UC 时，nt load 比 sc1 load 慢约 130 ns（约 670 对 540）。Lamport 的轮询读的是本地 inbox，暂不受影响。

---

## 2. 切分 H 是什么

- AR 部分每个 pack 互不相关。纯 Lamport AR 直接按 `pack / block_size` 分 block，完全不管行。
- RMSNorm 必须先拿到整行的 sum of squares 才能归一化，所以同一行的所有 pack 要汇总一次。
- block 同时决定两件事：一个 block 整个落在一个 CU 上；block 内汇总很便宜（LDS + `__syncthreads`），跨 block 只能走显存。

所以：一行只放一个 block，汇总便宜，但只能用一个 CU；一行拆成 k 个 block，能用 k 个 CU，但汇总要跨 block。

```
1 thread = 1 pack (16B = 8 x bf16);  H = 7168 -> 896 packs per row = 14 wave64

row   (block_size=896)  row r = [                  1 block, 14 waves                   ]  -> 1 CU
bs128 (block_size=128)  row r = [  b0  ][  b1  ][  b2  ][  b3  ][  b4  ][  b5  ][  b6  ]  -> up to 7 CUs
bs64  (block_size=64)   row r = [b0][b1][b2][b3][b4][b5][b6][b7][b8][b9][10][11][12][13]  -> up to 14 CUs
```

block_size 是每个 block 的线程数（= 每个 block 负责的 pack 数），不是 batch size；batch 方向的行数本文记作 M。下文的变体名 bs64 / bs128 / bs512 指 block_size = 64 / 128 / 512，row 指一行一个 block（block_size = H/8）。

要求 `(H/8) % block_size == 0`，所以 block 不会跨行（`custom_all_reduce.cuh` 约第 2442 行的注释）。grid = M × H/8/block_size。H=7168 时各变体的 block 数：

| M | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| row | 1 | 4 | 8 | 16 | 32 |
| bs128 | 7 | 28 | 56 | 112 | 224 |
| bs64 | 14 | 56 | 112 | 224 | 448 |

M=1 时，row 只用了 80 个 CU 里的 1 个。

---

## 3. 收益：把一行的读写分给多个 CU

### 3.1 每个线程要发多少条读写

每个线程负责一个 pack，每条读写都是 16 字节（dwordx4）。代码在 `custom_all_reduce.cuh` 约第 2508–2650 行：

| | TP2 | TP4 | TP8 |
|---|---|---|---|
| input load | 1 | 1 | 1 |
| push 到 peer（跨卡 store） | 1 | 3 | 7 |
| residual + weight load | 2 | 2 | 2 |
| 每轮轮询 inbox | 2 | 4 | 8 |
| residual_out + out store | 2 | 2 | 2 |
| rearm store | 1 | 3 | 7 |
| **合计（只轮询 1 轮）** | **9** | **15** | **27** |

轮询每轮都会把自己那一格也读一遍（没人写，结果也不用）。这是故意的：纯 AR kernel 约第 590 行的注释说，用分支跳过它会让每条 load 都等前一条。

### 3.2 全压在一个 CU 上是多少

H=7168 时一行 14 个 wave（H=8192 是 16 个，下面的数乘 16/14）。row 模式、M=1 时，这些全落在一个 CU 上：

| | TP2 | TP4 | TP8 |
|---|---|---|---|
| 一行的 dwordx4 指令数 | 126 | 210 | 378 |
| 单 CU 收费口时间（× 11.3 ns，即 16 cycle @1.42 GHz） | ~1.4 µs | ~2.4 µs | ~4.3 µs |
| 每轮轮询要同时在途的 load 数（上限 44） | 28 | 56 | 112 |

切开以后，每个 CU 只分到 1/k：

```
TP8, M=1, H=7168: dwordx4 instructions per CU (16 cycles each = 11.3 ns @ 1.42 GHz)

row   CU0       | push 98 | in/res/w 42 | poll 112 per round | st 28 | rearm 98 |  378  ~4.3 us
bs128 CU0..6    | push 14 | in/res/w  6 | poll  16 per round | st  4 | rearm 14 |   54  ~0.6 us
bs64  CU0..13   | push  7 | in/res/w  3 | poll   8 per round | st  2 | rearm  7 |   27  ~0.3 us
```

这几段都在关键路径上：
- push 发得慢，peer 就晚收到数据；
- 一轮轮询越长，数据到了之后越晚被发现；
- 末尾的 store 和 rearm 发不完，kernel 就结束不了。

TP8 的 row 还多一层问题：一轮轮询要发 112 条 load，路上最多只能放 44 条，至少要分 3 批，每批都要等一个完整往返。

### 3.3 和实测对照

纯 Lamport AR 不需要整行汇总，正好可以单独看"分给几个 CU"的影响（PROGRESS §5.2，1×7168，单位 µs）：

| | bs64（每行 14 个 block） | bs128（7 个） | bs512（2 个） |
|---|---|---|---|
| TP2 | 5.64 | 5.57 | 5.82 |
| TP8 | 7.20 | 7.63 | 9.18 |

- TP2 几乎不受影响；TP8 只用 2 个 block 时慢了 2 µs。H 相同，区别在于 TP8 每行的活约是 TP2 的 3 倍（27 对 9）。
- 融合版 TP8 1×7168：row 14.50 µs，bs128 10.26 µs，差 4.2 µs。只算收费口时间的估计是 4.3 − 0.6 ≈ 3.7 µs，再加上轮询分批、减去切分本身的代价（§3.4、§4），和 4.2 µs 很接近。
- TP2 下 row 在单个 CU 上一共只有约 1.4 µs 的收费口时间，其中不少能藏在等 peer 数据的时间里，所以切了也省不了多少。

### 3.4 微基准验证：TA 模型和切分的分界

`ta_split.cpp` 去掉了通信和 RMSNorm，只留访存：每个线程连发 L 条 dwordx4 load（全部发完才开始用），求和后 store 一条 dwordx4。一行 n=7168 个元素，即 896 个线程、14 个 wave。计时用 kernel 内的 span：所有 block 中最晚的结束时间减去最早的开始时间（`s_memrealtime`，10 ns 分辨率），取 300 次的中位数。

**变量**：

| 符号 | 含义 | 取值 |
|---|---|---|
| n | 一行的元素数（hidden size） | 7168 |
| n/512 | 一行的 wave 数（每线程 8 个元素，每 wave 64 线程） | 14 |
| I | **每个线程**发出的 dwordx4 访存指令数，load 和 store 都算 | 微基准里是 L+1；融合 kernel 见 §3.1（TP2/4/8 = 9/15/27） |
| N_CU | **最忙的那个 CU** 要发的 dwordx4 指令总数 = 这个 CU 上的 wave 数 × I | |
| 11.3 ns | 一条 dwordx4 占 TA 的时间，16 cycle ÷ 1.42 GHz | |
| T_base | 和 TA 无关的时间：block 分发、最后一条 load 的往返、等 store 完成 | 拟合值见下 |
| C | 把一行从 1 个 block 拆到多个 block 后 T_base 的增加量，也就是"多分 block 本身的代价" | 拟合值见下 |

**模型**：kernel 时间 ≈ T_base + N_CU × 11.3 ns

拟合结果：只有 1 个 block 时 T_base ≈ 0.85 µs；拆成 14 个 block（分布在 4 个 XCD）时 T_base ≈ 1.3 µs，所以 **C ≈ 0.45 µs**。C 是测出来的，不是推出来的。我推测它来自两方面：block 要依次分发到 4 个 XCD，kernel 要等最慢的那个 block 跑完。还没拆开验证。

**实测和模型对照**（M=1，单位 µs）。B=896 时一行 14 个 wave 都在 1 个 CU 上，N_CU = 14 × I；B=64 时一个 CU 只有 1 个 wave，N_CU = I：

| L | B=896 预测 | B=896 实测 | B=64 预测 | B=64 实测 | B=128 | B=448 |
|---|---|---|---|---|---|---|
| 1 | 0.85 + 0.32 = 1.17 | **1.16** | 1.3 + 0.02 = 1.32 | 1.37 | 1.37 | 1.29 |
| 4 | 0.85 + 0.79 = 1.64 | 1.72 | 1.3 + 0.06 = 1.36 | **1.45** | 1.49 | 1.61 |
| 8 | 0.85 + 1.42 = 2.27 | 2.40 | 1.3 + 0.10 = 1.40 | **1.53** | 1.61 | 1.93 |
| 27 | 0.85 + 4.43 = 5.28 | 5.40 | 1.3 + 0.32 = 1.62 | **1.89** | 2.05 | 3.53 |

- B=896 时 L 每加 1，时间增加约 0.16 µs，和 14 wave × 11.3 ns = 0.158 µs 吻合。这说明一行压在一个 CU 上时，TA 确实是瓶颈。
- L=1 时 row 反而最快：每个 CU 的 TA 时间只有 0.3 µs，省下的比不上 C。
- 换成 uncached 内存加 nt load，结论相同，数值略高（L=27：B=896 6.16，B=64 2.01）。

**切分什么时候划算**：一行从 1 个 block 拆成 k 个，最忙 CU 的指令数从 N 降到 N/k，省下 N × (1 − 1/k) × 11.3 ns，代价是 C。所以条件是

N × (1 − 1/k) × 11.3 ns > C，其中一行的 N = (n/512) × I

- **纯访存**（C ≈ 0.45 µs，k=14）：N ≳ 43。也就是**最忙的 CU 要发 40 条以上 dwordx4（约 40 KB）时，拆开才划算**。微基准的拐点在 L=2 到 4 之间（N = 42～70），和这个一致。
- **融合 AR+RMS**：C 还要加上跨 block 汇总的约 0.7 µs（§4.3），k=7，得 N ≳ 120。折算成 n：TP8 n ≳ 2300，TP4 ≳ 4100，TP2 ≳ 6800。这和 7168 下 TP2 用 row 更好、TP4/TP8 切了更好一致（§5）。但这个阈值偏乐观，因为真实 kernel 里一部分 TA 时间会藏在等 peer 数据的时间里，实际阈值会更高，精确的分界还要靠扫描。

**M 变大以后**（同一个微基准，单位 µs）：

| L=27 | B=64 | B=128 | B=448 | B=896（row） |
|---|---|---|---|---|
| M=1 | 1.89 | 2.05 | 3.53 | 5.40 |
| M=8 | 2.09 | 2.05 | 3.69 | 5.97 |
| M=80 | 7.65 | 7.49 | 7.57 | 7.59 |

M=80 时 row 已经每个 CU 一行，拆开不能再减少最忙 CU 的指令数，各种 B 一样，瓶颈变成整卡带宽。所以"M 能铺满 80 个 CU 时切分没用"是成立的，但这只是充分条件。融合 kernel 在 M≈8～16 就到分界了（§5），因为跨卡和显存的时间会先把 TA 时间盖住。

### 3.5 纯搬运 kernel（allgather 一类）：block_size=64 能不能再降 latency

`copy_bs.cpp`：用 dwordx4 的 load+store 搬 S 字节，每个 pack 一个线程，grid = S/16/B。扫了三个方向：本地→本地、本地→peer（push）、peer→本地（pull）。每格两个数，单位 µs：kernel 内的 span / hipGraph 里连续 launch 200 次时每个 kernel 的平均时间（含 launch 开销）。

| 本地→本地 | B=64 | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|---|
| 16KB | 1.31 / 2.44 | 1.31 / 2.39 | 1.35 / 2.39 | 1.13 / 2.45 | **0.84** / 2.45 |
| 64KB | **1.35** / 2.46 | 1.35 / 2.39 | 1.35 / 2.41 | 1.39 / 2.45 | 1.55 / 2.45 |
| 256KB | 1.43 / 2.48 | 1.43 / 2.42 | 1.43 / 2.42 | 1.51 / 2.47 | 1.55 / 2.47 |
| 1MB | 1.91 / 2.73 | **1.75** / 2.50 | 1.75 / 2.68 | 1.83 / 2.68 | 1.79 / 2.56 |
| 4MB | 4.25 / 5.42 | **3.47** / 4.00 | 3.51 / 4.16 | 3.75 / 4.27 | 4.11 / 4.89 |

- **跨卡方向**：≥256KB 时完全被 XGMI 带宽卡住（1MB 约 22 µs，约 47 GB/s），各种 B 都一样；push 的 span 不含 store 完成的时间，要看 graph 列。
- **结论：block_size=64 本身不会再降 latency。** 真正起作用的是最忙那个 CU 的指令数 N_CU，而不是 block 有多小。
  - 搬运时每 1 KB 只要 2 条指令（一 load 一 store），所以一个 CU 搬 X KB，TA 时间约 X × 22.6 ns。B=256 时一个 block 只搬 4 KB，不到 0.1 µs，再降到 64 也没有可省的了。
  - 只有 B 很大、block 又少时（例如 64KB 用 B=1024，4 个 block，每个 CU 16 KB），TA 才会多出约 0.2 µs，B=64～256 都能消掉这部分。
  - 数据极小时（16KB）block 越少越快：1 个 block 0.84 µs，16 个 block 1.31 µs。
  - block 太多也会变慢：4MB 用 B=64 有 4096 个 block，比 B=128 慢 0.8 µs。
  - **在 graph 里看端到端，≤1MB 时各种 B 相差不到 0.1 µs**。每个 kernel 约 2.4 µs 的 launch 开销占了大头，span 上的差别基本被它盖住了。

---

## 4. 代价：整行的 sum of squares 要跨 block 汇总

### 4.1 两种汇总路径

```
row (1 block per row): on chip
  14 waves: warpReduce(32 lanes) -> LDS shared[32] -> __syncthreads -> butterfly -> s_rms

split (k blocks per row): through memory
  each 32 lanes: warpReduce -> partial --nt store--> partials[round][group]   (rank-local, uncached)
  each block   : lanes 0..27 spin on partials[row*28 + lane] until != 0xFFFFFFFF
                 -> butterfly -> s_rms                     (all k blocks repeat this)
```

代码在 `custom_all_reduce.cuh` 约第 2583–2626 行。

### 4.2 跨 block 汇总贵在哪

1. **多一次显存往返。** partial 先 store 出去，别的 block 再 load 回来，走的是 uncached 内存。按 §1.5，UC 的写不留在 L2，要一直写到显存，并把其它 L2 里的旧副本失效；读的一方再从显存取回来。所以这一趟往返和"几个 block 在不在同一个 XCD"基本无关，同一个 XCD 也省不掉。要想让同一个 L2 帮上忙，需要同时做两件事：(a) 重新映射 block ID，让同一行的 block 满足 `blockIdx % 4` 相同，从而落在同一个 XCD；(b) partial 改用可缓存的内存。但内存模型里没有"XCD 作用域"，这个做法依赖分发规律，没有任何保证。
2. **vmcnt 按序完成。** gfx9 上记录"还有几条读写没完成"的计数器（vmcnt）同时计 load 和 store，而且按发出顺序完成。所以轮询 partial 的那条 load，要等这个 wave 前面所有读写都完成才能用，其中包括跨卡 push、residual_out store 和 partial store。
3. **要等最慢的 block。** 一行的 k 个 block 都要等其中最慢的那个发布 partial，k 越大越容易被拖慢。
4. **重复轮询。** k 个 block 每个都要把这一行的 28 个 partial 读一遍。

### 4.3 实测

- **负载轻时约 0.7 µs**：TP2 1×7168 下 bs128 是 8.90 µs，row 是 8.19 µs。TP2 切分几乎没有收益，所以这个差基本就是交换的代价。
- **负载重时会被放大**：M=32 时所有切分版都比 row 慢，TP8 最明显（32×7168：bs64 32.32，row 26.26；32×8192：bs64 35.11，row 29.16）。这时 row 已经和纯 AR 差不多（32×7168 是 26.26 对 26.67），说明单个 CU 的那约 4.3 µs 已经被等跨卡数据的时间盖住了；剩下的约 6 µs 主要来自交换，以及交换导致 epilogue 没法被藏住。
- **另一种做法也试过**：把 norm 放到第二个 kernel（`lamport_ar+norm`），等于用 kernel 边界来做整行同步，实测在所有 case 下都比融合版慢（NOTES §5.2）。
- **还没查清**：§4.2 的四处里哪一处被放大得最多，还没拆开测，这就是 NOTES §6.1 那条待办。

---

## 5. 放到 TP × M 上看

```
fastest Lamport variant (fused, MI308X, H = 7168 and 8192)

         M=1     M=2     M=4     M=8     M=16    M=32
TP2      row     row     row     row     row     row
TP4      split   split   tie     tie     row     row*
TP8      split   split   split   split*  row*    row*

split = bs64/bs128 (mostly bs128)   tie = within 0.4 us
*     = car_fused beats every Lamport variant here
```

- **往右看（M 变大）**：row 模式下有活干的 CU 数就是 M，整体发射能力跟着 M 增长。瓶颈转到跨卡链路和显存上，单个 CU 的那几 µs 被等待时间盖住，切分就只剩代价了。M ≥ 80 时 row 已经每个 CU 一行，切分在原理上就不可能再有收益（§3.4 微基准 M=80 各种 B 一样）；实际的分界在 M≈8～16，比 80 早得多。
- **往下看（TP 变大）**：每行的活随 TP 线性增长（9 → 15 → 27），一个 CU 扛不住的区间更宽，所以分界线往右移。TP2 在任何 M 下都不该切；TP4 在 M=4–8 打平；TP8 到 M=8 还是切了更快。

这和 NOTES §3.3 里的两个条件说的是同一件事："一行的活够多"指单个 CU 的收费口时间，"M 太小铺不满"指不切时有几个 CU 在干活。

---

## 6. bs64 和 bs128

- 纯 AR 里 bs64 略好，因为分得更散。
- 融合版里多数情况 bs128 更好：TP8 1×7168 是 10.26 对 10.86，TP4 是 9.19 对 9.68。

原因是 partial 的个数和 block_size 无关（每 32 lane 一个，一行固定 28 或 32 个），变化的是参与汇总的 block 数。bs64 每行 14 或 16 个 block，每个都要轮询全部 partial、都要等最慢的那个；bs128 只有 7 或 8 个。7 个 CU 已经把单 CU 的收费口时间压到约 0.6 µs，再翻一倍只能多省约 0.3 µs，抵不过参与 block 翻倍带来的同步开销。§3.4 的纯访存微基准里 bs64 和 bs128 本来就只差约 0.15 µs（L=27：1.89 对 2.05），融合版多了跨 block 汇总，差距就倒过来了。

---

## 7. 对调度的含义

- **门槛**：和 NOTES §6.3 的初步结论一致。TP2 用 row；TP4、TP8 在小 M 时用 bs128；M 到 8–16 以后用 row，或者退回现有的 fused。
- **实现约束**：block_size 在 `OneShotRmsState` 的生命周期里必须固定（`custom_all_reduce.cuh` 约第 2451–2453 行）。所以如果要按 M 在 row 和 bs128 之间切换，需要建两个 state，也就是两份 inbox（TP8 每个 rank 约 15 MiB）。
- **降低切分代价的方向都在交换上**：同 XCD 交换（前提见 §4.2 第 1 条：partial 要放在可缓存内存上）、用标量 load 轮询 partial（NOTES §6.2），以及 gfx1250 上的 cluster。标量 load 走 SQC，不经过 TA，用 lgkmcnt 而不是 vmcnt 计数，所以既避开了 vmcnt 的按序等待，也不占 TA 的 16 cycle。但标量缓存同样不保证一致，要带 sc/glc 位或在每轮之前失效。
- **轮询指令的写法**：现在用的是 nt load（§1.5 实测不命中 L1，所以能看到新数据），但这是实测行为，不是文档保证。改用 `__hip_atomic_load(relaxed, agent/system)`（编译成 sc1 / sc0 sc1）是内存模型明文保证的写法，本地延迟和 nt 一样。

---

## 8. 待办

1. **在真实 kernel 里验证 §3 的单 CU 模型**：微基准已经验证（§3.4），真实 kernel 还没有。在 TP8 1×7168 下对比 row 和 bs128 的 TA busy 和 LFIFO stall 计数器（白皮书 §3.4.7 的 `perf_tcp_lfifo_stall_cycles`）。如果模型对，row 那个 CU 的 TA 应该几乎一直是满的。
2. **拆交换的代价**（NOTES §6.1）：M 大时是 §4.2 的哪一环被放大。
3. **附录 B 的两个问题**：决定要不要修，或者报给对应模块的 owner。
4. **拆 C**（§3.4）：一行拆成 14 个 block 的约 0.45 µs 固定代价，是 block 分发到 4 个 XCD 的时间，还是等最慢 block 的时间。可以记录每个 block 的开始、结束时间来看分布。
5. **轮询指令**（§7）：把 nt load 换成 `__hip_atomic_load(relaxed, agent)`，确认正确性和延迟不变。顺便查一下读 peer UC 时 nt 比 sc1 慢约 130 ns 的原因（§1.5）。

---

## 附录 A：custom allreduce 1-shot / 2-shot 数据流（TP4）

按 `csrc/include/custom_all_reduce.cuh` 画：1-shot 是 `cross_device_reduce_1stage`（约第 378 行），2-shot 是 `cross_device_reduce_2stage`（约第 824 行）。

### A.1 TP4 怎么分流

`use_new=True`、XGMI 全互联、bf16（约第 5420–5514 行）：

```
numel <= 28672  (<= 56KB)  -> ar_ll_gfx9                   LL
numel <= 32768  (<= 64KB)  -> ar_ll128_unroll2             LL128
bytes  <  160KB            -> cross_device_reduce_1stage   1-shot
else                       -> cross_device_reduce_2stage   2-shot
                              bytes % 64 != 0  -> _naive
                              packs % 512 != 0 -> has_tail
                              write_mode: TP8 + gfx942 + >4MB only
512 threads/block, 1 pack = 16B = 8 x bf16
```

### A.2 记号和 buffer

- rank0..3 的输入记为 A/B/C/D，`Y = A+B+C+D`，S 是消息字节数。
- **XGMI**：GPU 之间的直连链路。图里 `xgmi` 表示跨卡读，`local` 表示读本卡。
- `reg_inp`（IPC，peer 可以直接访问）：kernel 读的输入，即 `_input_dp->ptrs[r]`。eager 模式先 `hipMemcpyAsync(inp -> reg_inp)`；graph 模式下 inp 本身已经注册。
- `meta`（IPC）：`[Signal{start[80][8], end[80][8], _flag[80]} | tmp ...]`，即 `sg.signals[r]`。`tmp = (Signal*)sg + 1`，只有 2-shot 用。
- `out`：本地 buffer，只有本 rank 写。
- 两种都是"拉"：数据靠 load peer 的 IPC 内存，往 peer 写的只有 flag。

### A.3 1-shot

```
              rank0            rank1            rank2            rank3
           +--------+       +--------+       +--------+       +--------+
 reg_inp   |   A    |       |   B    |       |   C    |       |   D    |   S bytes each
           +--------+       +--------+       +--------+       +--------+
 ==================== start_sync (per block b, all 4 ranks) ====================
           local A          local B          local C          local D
           xgmi B C D       xgmi A C D       xgmi A B D       xgmi A B C     3*S in / rank
               |                |                |                |
               v                v                v                v
            A+B+C+D          A+B+C+D          A+B+C+D          A+B+C+D       fp32, order 0,1,2,3
               |                |                |                |
           +---v----+       +---v----+       +---v----+       +---v----+
 out       |   Y    |       |   Y    |       |   Y    |       |   Y    |   local store
           +--------+       +--------+       +--------+       +--------+
 ===================== end_sync<final=true> (relaxed) ==========================
```

- 每个 rank 各自算整份 Y，所以 4 个 rank 的求和顺序必须相同（都是 A+B+C+D），结果才逐位一致。
- 末尾的 barrier 不能省。它保证所有 rank 都读完了彼此的 reg_inp 才返回，否则跑得快的 rank 下一次调用的 memcpy 会改掉别人还在读的输入。

block 内部（代码里叫 `warp_id`，实际一组是 128 线程，即 2 个 wave64）：

```
   A[idx] @rank0 --> group0 (thr   0..127) --+
   B[idx] @rank1 --> group1 (thr 128..255) --+--> LDS tmp_smem[buf][4*128]  (16B packs)
   C[idx] @rank2 --> group2 (thr 256..383) --+     |
   D[idx] @rank3 --> group3 (thr 384..511) --+     v
                                              group0: Y[idx] = A+B+C+D (fp32) --> out[idx]
```

- group g 固定读 rank g，不旋转。在 rank r 上，group r 是本地读，另外 3 组跨卡读。
- TP4 走 1-shot 时 S < 160KB，也就是不到 10240 个 pack（= 80 blocks × 128），所以每个 block 只跑 1 轮。kernel 里的 LDS 双缓冲（group0 规约当前 buf 时，其它组预取下一轮）在 TP4 和 TP8 上都用不到，只有 TP2 的大消息才会跑多轮。

### A.4 2-shot

stage1 是 reduce-scatter（RS，每个 rank 规约自己负责的那 1/4），stage2 是 all-gather（AG，每个 rank 把 4 段结果都拉回来）。

```
               rank0            rank1            rank2            rank3
           +-----------+    +-----------+    +-----------+    +-----------+
 reg_inp   |A0 A1 A2 A3|    |B0 B1 B2 B3|    |C0 C1 C2 C3|    |D0 D1 D2 D3|   S/4 per chunk
           +-----------+    +-----------+    +-----------+    +-----------+
 ======================= start_sync (per block b, all 4 ranks) ======================
 stage1    local A0         local B1         local C2         local D3
 (RS)      xgmi B0 C0 D0    xgmi C1 D1 A1    xgmi D2 A2 B2    xgmi A3 B3 C3    3*S/4 in / rank
                 |                |                |                |
           +-----v-----+    +-----v-----+    +-----v-----+    +-----v-----+
 tmp       |     Y0    |    |     Y1    |    |     Y2    |    |     Y3    |   Yk = Ak+Bk+Ck+Dk
           +-----------+    +-----------+    +-----------+    +-----------+
 ======================== end_sync (release / acquire) ==============================
 stage2    local Y0         local Y1         local Y2         local Y3
 (AG)      xgmi Y1 Y2 Y3    xgmi Y2 Y3 Y0    xgmi Y3 Y0 Y1    xgmi Y0 Y1 Y2    3*S/4 in / rank
                 |                |                |                |
           +-----v-----+    +-----v-----+    +-----v-----+    +-----v-----+
 out       |Y0 Y1 Y2 Y3|    |Y0 Y1 Y2 Y3|    |Y0 Y1 Y2 Y3|    |Y0 Y1 Y2 Y3|   local store
           +-----------+    +-----------+    +-----------+    +-----------+
 - - - - - - - - - - - - - - - - - - no end barrier - - - - - - - - - - - - - - - - - -
```

以 chunk 0 为例（owner 是 rank0），其它 chunk 只是换一个 owner：

```
   A0 @rank0 (local) --+
   B0 @rank1 (xgmi)  --+
   C0 @rank2 (xgmi)  --+--> rank0: Y0 -> tmp0 --+--> rank0 out[chunk0]   local
   D0 @rank3 (xgmi)  --+                        +--> rank1 out[chunk0]   pulled via xgmi
                                                +--> rank2 out[chunk0]   pulled via xgmi
                                                +--> rank3 out[chunk0]   pulled via xgmi
             stage1: reduce-scatter                  stage2: all-gather
```

block 内部：group g 对应 rank (r+g)%4，所以 group0 永远读自己。

```
 stage1 on rank r (owner of chunk r):
   reg_inp@rank r   [chunk r] --> group0 --+
   reg_inp@rank r+1 [chunk r] --> group1 --+--> LDS --> group0 sum (fp32) --> tmp@rank r
   reg_inp@rank r+2 [chunk r] --> group2 --+
   reg_inp@rank r+3 [chunk r] --> group3 --+
 ------------------------------ end_sync ------------------------------
 stage2 on rank r:
   tmp@rank r                 --> group0 --> out[chunk r  ]
   tmp@rank r+1               --> group1 --> out[chunk r+1]
   tmp@rank r+2               --> group2 --> out[chunk r+2]
   tmp@rank r+3               --> group3 --> out[chunk r+3]
 (rank ids mod 4; group0 = local, group1..3 = xgmi)
```

- rank r 的求和顺序是 r, r+1, r+2, r+3，每段的顺序因此不同。但每段只由 owner 算一次，其它 rank 直接拷贝，所以各 rank 的结果仍然逐位一致。
- 两个 stage 必须用同一套 idx→block 映射。原因是 barrier 按 block 走：block b 只等各 rank 的 block b（约第 911 行的注释）。
- grid = ceil(packs/4/128)。S=160KB 时是 20 个 block，S ≥ 640KB 时封顶 80 个，更大的消息每个 block 循环多轮。
- 没有末尾 barrier：过了中间那次 barrier 后，没有 rank 再读 reg_inp；tmp 要到下一次 allreduce 的 stage1 才会被改写，而写之前要先过那一次的 start_sync，到那时 peer 上一次的 kernel 已经整个跑完（同一 stream 串行执行）。这个前提对 LL 路径不成立，见附录 B.1。

### A.5 barrier（`start_sync` 约第 162 行，`end_sync` 约第 209 行）

```
 per block b (ROCm path):
   flag = self->_flag[b] + 1
   thread t (t < 4):  store flag -> signals[t]->start[b][rank]    // remote write
                      spin while self->start[b][t] < flag         // wait for rank t
   __syncthreads();   self->_flag[b] = flag                       // only grows, never reset

   Signal.start[b] on rank0:   [ slot0 | slot1 | slot2 | slot3 ]
                                   ^       ^       ^       ^
   written by (block b of):      rank0   rank1   rank2   rank3

 end_sync: same handshake on Signal.end[b][*]
```

- `start_sync` 用 relaxed，因为开头还没有数据需要发布。
- 2-shot 中间的 `end_sync` 先做 `__syncthreads`，再用 release 写、acquire 读，保证 stage1 写进 tmp 的数据对 peer 可见。
- 1-shot 末尾的 `end_sync<final=true>` 用 relaxed，只表示"大家都读完了"。

### A.6 对比（TP4）

| | 1-shot | 2-shot |
|---|---|---|
| 适用大小 | bf16：64KB < S < 160KB（fp32 没有 LL，S < 160KB 都走这里） | S ≥ 160KB |
| barrier | start + end（final，relaxed） | start + 中间一次（release/acquire），没有末尾 |
| 先后执行的远端读阶段 | 1 | 2（RS 做完才能 AG） |
| 每个 rank 从 XGMI 读入 | 3·S | 2 × 3·S/4 = 1.5·S |
| 每条链路单方向 | S | S/2 |
| 每个 rank 的规约量 | S（4 个 rank 重复算同一份） | S/4 |
| 额外 buffer | 无 | tmp，S/4 |
| 结果逐位一致的原因 | 4 个 rank 求和顺序相同 | 每段只算一次，其它 rank 拷贝 |

一般地，N 张卡时 1-shot 每个 rank 读入 (N−1)·S，2-shot 读入 2(N−1)/N·S。TP4 是 3S 对 1.5S，TP8 是 7S 对 1.75S，所以 TP8 切到 2-shot 的点更低（80KB）。小消息主要受延迟限制，1-shot 少一个远端读阶段，所以小的走 1-shot，大的走 2-shot。

---

## 附录 B：顺带发现的两个问题（读代码推断，没有运行验证）

### B.1 2-shot 后紧跟 LL 可能有竞争

2-shot 能省掉末尾 barrier，前提是下一个写 tmp 的 kernel 会先过 start_sync，但 LL 路径不过 barrier：
- LL scratch 的起始地址就是 `(Signal*)sg + 1`（`custom_all_reduce.cuh` 约第 4694–4696 行），和 2-shot 的 tmp 是同一块内存。
- `ar_ll_gfx9` 直接把数据写到 peer scratch 的 `bank*4*slot + rank*slot` 处（约第 4188 行）。rank0 用 bank 0 时，写的正好是 peer tmp 开头的 2×msg 字节。
- 所以如果一次 2-shot 后面紧跟一次 LL，rank0 可能已经在改 peer 的 tmp，而慢一点的 rank 还在 stage2 读它。
- 模型里两次 allreduce 之间有计算，这个时间窗口很小；连续调用、大小交替的 microbench 更容易碰到。

### B.2 fp16 在 graph 模式下不走 LL/LL128

当 `reg_inp_ptr == 0`（输入本身已注册，即 graph 模式）时 `is_broadcast_reg_outptr = true`，fp16 会跳过 LL/LL128，≤64KB 的消息也走 1-shot。bf16 那一支没有传这个参数，所以不受影响（`csrc/kernels/custom_all_reduce.cu` 第 260 行对比第 269 行）。
