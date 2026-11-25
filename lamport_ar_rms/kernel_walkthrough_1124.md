# Lamport AR + RMSNorm（含 FP8 量化）优化总结

> - **代码**：`/app/aiter/csrc/include/custom_all_reduce_gfx1250.cuh`
> - **版本**：旧版 = main `3845cd612`，新版 = PR #8559 `3b15a6366`。行号写作 `新:1497`、`旧:816`
> - **汇编**：从两版编译出的 `module_custom_all_reduce_gfx1250.so` 反汇编得到（TP4、bf16），只截关键行，`...` 表示省略。行尾 `; ...` 是我加的注释。获取方法见第 14.2 节
> - **测量**：4×gfx1250，每次 10 个调用录成一个 graph，replay 20 次取中位数，取 4 个 rank 里最慢的那个。单位 us。没有特别说明的都是 TP4、bf16 输出
> - 上一版逐行讲解备份在同目录 `kernel_walkthrough_1124_line_by_line.md`

---

## 0. 怎么读这份文档

- **第 1 节 背景**：kernel 算什么、跨卡协议怎么工作、时间花在哪。后面每个 tip 都要用到，建议先看。
- **第 2 节 总览**：全部 tip 按重要性排序，各自能提升多少。
- **第 3~13 节**：每节一个 tip，固定按下面的顺序写：
  1. 提升多少
  2. 原来哪里慢、怎么发现的
  3. 怎么改、为什么这样改
  4. 原来的代码
  5. 现在的代码
  6. 执行过程（一步一步）
  7. 指令拆解（先说指令本身的意思，再说这里用它做什么、目的是什么）
  8. 通用性和经验
- **第 14 节 方法**：这些问题是用什么手段找到的。比单个 tip 更值得学。
- **第 15 节 试过没用的**。
- **第 16 节 术语表**：文中第一次出现的术语都可以在这里查。

**关于"提升多少"**：各个 tip 是在开发的不同阶段测的，基线不一样，**数字不能直接相加**。每个数字都注明了在哪个版本、哪个形状上测。最终的整体效果：TP4 比 main 中位快 25%（bf16 输出）和 33%（FP8 输出），每个测过的形状都不比 Gluon 慢。

---

## 1. 背景

### 1.1 kernel 算什么

每张卡上有一份 `input[M][hidden]`（bf16），一次 kernel 完成：

```
ar      = 4 张卡的 input 按卡号顺序相加（fp32 累加），舍入成 bf16
res_out = ar + residual（fp32 加）                     → 写 residual_out
out     = res_out × rsqrt(mean(res_out²) + eps) × weight
          bf16 输出：写 output
          FP8 输出：求整行 |out| 的最大值 amax
                    scale = amax / fp8_max（e4m3fn 是 448）
                    out × (1/scale) → 截到 ±fp8_max → 转 FP8，写 output 和 scale_out
```

**约束**：结果必须和 `ar_rmsnorm_gfx1250`（下称 car_fused）逐位一致，所以每一步的运算顺序都不能改。很多 tip 的写法都受这个约束，例如浮点加法的顺序必须保持。

### 1.2 硬件单位

| 名词 | 含义 |
|---|---|
| CU | 计算单元。gfx1250 每卡 256 个。一个 block 只在一个 CU 上运行 |
| wave | 32 个线程，同时执行同一条指令（wave32）。wave 内的线程叫 lane，编号 0~31 |
| block | 若干个 wave。同一 block 的 wave 共享 LDS，可以做 block 同步 |
| LDS | CU 内部的共享内存，访问几十个时钟；显存要几百到上千个时钟，跨卡更慢 |
| VGPR / SGPR | 向量寄存器（每个 lane 一份）/ 标量寄存器（整个 wave 共用一份） |
| VALU / SALU | 向量运算单元 / 标量运算单元。一个 CU 上所有 wave 共用 SALU |
| 包（pack） | 16 字节 = 8 个 bf16，一次读写的单位 |
| row_packs | 一行的包数 = hidden / 8。例如 6144 → 768 |
| 组（group） | 连续 32 个包，正好一个 wave 处理，一个轮次计数器管一组 |

### 1.3 跨卡协议：Lamport 收件箱

```
每张卡有一块 IPC 显存（别的卡可以直接写），初始全部填 0xff：
    inbox[缓冲 0..2][发送方 0..3][包 0..slot-1]，每格 16 字节

第 r 轮：卡 A 把包 p 写到每张别的卡 B 的 inbox[r%3][A][p]
         卡 B 反复读自己的 inbox[r%3][*][p]，读到 0xffff 就是还没到
用完后：把 (r+2)%3 号缓冲（上一轮用过的）重新写成 0xffff，叫"回填"（rearm）
```

- **哨兵 0xffff**：bf16 的 0xffff 是一个 NaN。发送前把数据里的 0xffff 改成 0x7fff（也是 NaN），所以真实数据永远不会被误认成"没到"。
- **为什么是 3 块缓冲**：最快的卡最多领先一轮。第 r 轮正在用 r%3，领先的卡可能已经在写 (r+1)%3，只有 (r+2)%3 没人用，可以回填。
- **轮次计数器**：每组包一个，存在本卡显存里。所有卡的调用顺序相同，所以同一组包的计数器在每张卡上都一样，不需要卡间同步。
- **好处**：没有卡间 barrier，数据一到就能用。**代价**：每次要回填，接收方要反复轮询。

### 1.4 时间花在哪：旧版的一次调用

旧版每个线程负责一个包，依次做：

```
① 读本卡 input → 写到另外 3 张卡（跨卡写）
② 读 residual、weight
③ 轮询自己的收件箱，直到 3 张卡的数据都到了
④ 4 份相加、舍入、加 residual、算本线程的平方和
⑤ 整行求和（block 同步）→ inv_rms
⑥ 乘 weight，写输出（FP8 还要再求一次整行最大值）
⑦ 回填，计数器 +1
```

小 M 时，旧版有两种排法：row（一行一个 block）和 cluster（一行拆给 2~16 个 block）。下面是 6144×1 在 kernel 内打的时间戳，单位 us，从第一个 wave 开始算（优化前测）：

| 阶段结束时刻 | row（一行 1 个 CU） | cluster（一行 3 个 CU） |
|---|---|---|
| ① 发送完 | 0.80 | 0.88 |
| ③ 轮询完 | 5.04 | 3.60 |
| ⑤ 规约完 | 6.16 | 5.80 |
| 全部写完 | 6.64 | 6.24 |

从这张表能看出两个大头：

1. **等数据（①→③）**：占 3~4 us。数据从各卡发出、到达得越早，这段越短。
2. **整行规约（③→⑤）**：row 1.1 us，cluster 2.2 us。按计算量算，这一步只需要几十纳秒。

几乎所有 tip 都在打这两个大头：

| 大头 | 对应的 tip |
|---|---|
| 等数据 | #1 发送和接收分开（split）、#4 参数预载、#5 发送后等一次参数读、#8 BYPASS 读写、#9 只轮询 3 张卡 |
| 规约 | #2 同步只等 LDS、#6 FP8 amax、#7 DPP |
| M 大时 | #3 一个线程负责多个包（PACKS） |

### 1.5 新版的两种模式

```
row 模式（M 较大，或 hidden ≤ 1024）：每行一个 block，自己发、自己收
    读本卡数据 → 发给 3 张卡 → 等一次参数读 → 读 residual/weight → 轮询 → 计算 → 回填

split 模式（TP4 时 M ≤ 64 且 hidden > 1024）：
    grid = [发送块 0 .. senders-1][接收块：每行一个]
    发送块：读本卡数据 → 发给 3 张卡 → 回填 → 退出
    接收块：读本卡数据、residual、weight → 轮询 → 计算 → 写输出
```

---

## 2. 总览

按重要性排序。重要性综合考虑了收益大小和覆盖的形状范围。

| # | tip | 提升（形状 / 在哪个版本测的） | 覆盖范围 | 能否用到别处 |
|---|---|---|---|---|
| 1 | **发送和接收分到不同的 block（split）**，删掉 cluster | 6144×1：row 模式中最快的 7.77 → 5.95（−23%），8192×1 8.39 → 6.12（−27%）。对 main（cluster）：6144×1 8.19 → 6.05 / 最终版 | TP4 M ≤ 64，hidden > 1024 | 高：凡是"一个 CU 既发又收"的通信 kernel |
| 2 | **block 同步只等 LDS，不等跨卡写的确认** | 2048×1 7.05 → 5.65 / POC 阶段 | 所有形状 | 高：发完远端写、紧接 block 同步的 kernel |
| 3 | **一个线程负责多个包（PACKS）** | 8192×128 15.96 → 13.21（−17%），6144×256 18.11 → 15.88（−12%），4096×128 10.12 → 8.89（−12%） / 最终版 | row 模式（TP4 时 M > 64） | 中：大 block、一行一个 block 的 kernel |
| 4 | **kernel 参数全部放进预载区** | split 6144×1：bf16 6.59 → 6.10，FP8 7.08 → 6.43 / 开发中期 | 所有形状 | 高：参数多、延迟敏感的小 kernel |
| 5 | **row 模式发完后等一次参数读** | 6144×1 10.40 → 7.52，8192×1 11.53 → 8.25，2048×1 6.01 → 5.43 / 移植版 | row 模式 | 低：经验结果，机制没证实 |
| 6 | **FP8 amax：按整数比较，LDS 原子 max** | FP8 求 amax 的成本约 0.45 → 0.2 us / 开发后期 | FP8 输出 | 高：对非负、无 NaN 的 float 求最大值 |
| 7 | **wave 内求和用 DPP，跨 wave 只同步一次** | 2048×1：bf16 5.65 → 5.51，FP8 6.17 → 5.73 / POC 阶段 | 所有形状 | 中：wave32 架构 |
| 8 | **跨卡读写改用 buffer 指令 + BYPASS** | 约 1~3%，接近噪声 / 移植版 | 所有形状 | 中 |
| 9 | **轮询只读另外 3 张卡** | 没有单独测准；轮询读减少 1/4（TP4）、1/2（TP2） | 所有形状 | 高：去掉无用访存 |
| 10 | **host 按形状自动选模式** | 比"每个形状取 6 种固定配置里最快的"平均只慢 0.1%（TP4）/ 1.4%（TP2） | — | 方法可推广 |
| 11 | **禁止乘加合并 `fp contract(off)`** | 不提速，保证逐位一致 | — | 高：凡是要求逐位一致的 kernel |

**FP8 路径的总效果**：同形状下，FP8 输出比 bf16 输出多花的时间，TP4 中位数从 0.89 us 降到 0.25 us。这是 #6、#7、#4（scale_out、bf16_output 进预载区）和删掉 cluster（#1）的合计。

**关于 #5 的排位**：它的数字很大，但基线是"所有参数都已预载"的移植版。main 原版其实已经"碰巧"有这一步等待（见第 7 节），所以相对 main 它不是净收益，而是补回了 #4 带走的东西。它排在后面，是因为机制没有证实，不能放心推广。

---

## 3. Tip 1：发送和接收分到不同的 block（split），删掉 cluster

### 3.1 提升多少

最终版代码，同一次测量里比较各模式（TP4，bf16，us）：

| hidden×M | row 模式中最快的 | cluster（main） | split | split 比 row 快 |
|---|---|---|---|---|
| 4096×1 | 6.21（PACKS=2） | 7.64 | 5.68 | −9% |
| 6144×1 | 7.77（PACKS=4） | 8.19 | 5.95 | −23% |
| 8192×1 | 8.39（PACKS=4） | 8.45 | 6.12 | −27% |
| 6144×16 | 8.26（PACKS=4） | 8.75 | 6.64 | −20% |
| 6144×64 | 9.51（PACKS=4） | 10.41 | 8.33 | −12% |

cluster 那一列是 main 的整体耗时（main 在这些形状上走 cluster），里面还混着别的优化缺失，只作参考。

**这是收益最大的 tip。** 6144 小 M 原来是 Gluon 最占优势的形状（Gluon 6.15 us），没有 split 时怎么调都追不上。

### 3.2 原来哪里慢、怎么发现的

**现象**：小 M、大 hidden 时，一行只有一个 block，也就是只有一个 CU 在干活。以 6144×1 为例，这个 CU 要：

```
发 768 个包 × 3 张卡 = 2304 次跨卡写
然后反复读 768 × 3 = 2304 个格子，直到数据都到
最后再写 2304 次回填
```

一个 CU 的访存通道同一时间能处理的请求是有限的。2304 次写只能排着队发出，后面的读也排在同一条通道上。

**怎么确认的：消融实验**（第 14.1 节）。6144×1 上去掉不同的环节，看总时间：

| 变体 | us |
|---|---|
| 单个 CU，只发送（跳过轮询和规约） | 6.28 |
| 单个 CU，发送 + 轮询（跳过规约） | 7.02 |
| 3 个 CU 的 cluster，发送 + 轮询（跳过规约） | 6.31 |
| 对照：AR+add kernel（同样的数据量，12 个 64 线程的 block） | 5.57 |

结论：

- 光是一个 CU 把 2304 次写发完，就要 6.28 us，比 Gluon 整个 kernel 还慢。**瓶颈在"一个 CU 发得太慢"**，不在计算。
- 把同样的事分给更多 CU（cluster、AR+add），数据到得明显更快。

**cluster 为什么没解决问题**：cluster 把一行拆给 3~16 个 block，发送确实分散了，但平方和也被拆到了多个 block 里。block 之间没有共享的 LDS，部分和只能经过显存交换：

```
每个 wave 把部分和写进显存
→ 等本 wave 所有写都被确认（包括刚才的跨卡写！）
→ cluster 同步
→ 作废本地 cache，再从显存读所有部分和
→ 串行地一个一个加
```

上面的时间戳表里，cluster 的规约要 2.2 us，是 row 的 2 倍。也就是 cluster 用规约变慢，换来了发送变快。

**Gluon 的做法给了启发**：Gluon 把程序分成两类，前面的只负责发，后面每行一个程序只负责收和做 norm。

### 3.3 怎么改、为什么这样改

**核心思路：需要协作的部分留在一个 block 里，不需要协作的部分分散出去。**

- 发送：每个包独立，不需要和别的包协作 → 分给很多个"发送块"，每块 256 个包。
- 规约：整行要求和，需要协作 → 留在一个"接收块"里，用 LDS，不走显存。

```
grid = [发送块 0][发送块 1][发送块 2] ... [接收块 行0][接收块 行1] ...
         每块发 256 个包，发完顺便回填      每行一个，只轮询、计算、写输出
```

为此要解决两个问题：

**问题 1：轮次计数器会被抢。** 原来同一个线程先读计数器、再加 1，没有竞争。现在发送块和接收块是不同的 block，执行顺序不确定。如果共用一个计数器，可能出现：

```
发送块读到 r → 发送 → 写回 r+1
                              接收块这时才读，读到 r+1 → 去错误的缓冲里等 → 永远等不到
```

解决：计数器拆成两份，tx（发送块用）和 rx（接收块用），各读各写。一次调用结束时，每组包的 tx 和 rx 都恰好加了 1，所以始终相等。其他不拆分的 Lamport kernel（1-shot、2-shot、AR+add、all-gather）每次两份都写，几种 kernel 交替调用也不会乱。

**问题 2：回填交给谁。** 原来是接收方算完之后回填。现在交给发送块，发完立刻回填。为什么可以这么早：

- 要回填的是 (r+2)%3 号缓冲，也就是上一次调用用过的那块。
- 上一次调用的 kernel 已经结束（kernel 之间有先后），本卡不会再读它。
- 别的卡最多领先一轮，只会写 (r+1)%3，不会写 (r+2)%3。

好处：接收块尾部少了 2304 次写，回填也分散到了多个 CU。

**发送块每块 256 个包**，是在 32/64/128/256 和"一行一个发送块"之间测出来的：

- 太小（32）：发送块和接收块是同一次 launch，block 大小一样（接收块的 768 线程），发送块只用其中 32 个线程，其余直接退出，浪费调度。
- 太大（一整行）：又回到一个 CU 发一整行，6144~8192 时 9.5~12.7 us。

**什么时候不用 split**：

- M 大了，行数本身就能铺满 CU。TP4 下 M=64、hidden 6144 时，发送块 192 个 + 接收块 64 个 = 256 个，正好等于 CU 数；M 再大，block 就比 CU 多了，split 没有优势。
- hidden ≤ 1024 时一行只有 128 个包，一个 CU 发得过来。

**cluster 整个删掉**：最终测量里 split 在 cluster 原来覆盖的每个形状上都更快。

### 3.4 原来的代码

host 端选 cluster（`旧:2311`）：

```cpp
// 行数少于 CU 数，且一行超过 256 个包时，把一行拆给一个 cluster
int cluster_threads(int64_t rows, int row_packs) const
{
    if(!kLamportClusterLaunch || rows >= num_cus || row_packs <= kLamportMinClusterRowPacks)
        return 0;                                  // 0 = 一行一个 block
    for(int bs = kLamportClusterMaxThreads; bs >= kLamportWave; bs /= 2)
    {
        if(row_packs % bs != 0)
            continue;
        const int blocks = row_packs / bs;
        return blocks <= kLamportMaxClusterBlocks ? bs : 0;
    }
    return 0;
}
```

kernel 里 cluster 的规约（`旧:1618`）：

```cpp
float part = square_sum;
for(int off = kLamportWave / 2; off > 0; off >>= 1)
    part = part + __shfl_xor(part, off);                    // wave 内求和
const int lane = threadIdx.x % kLamportWave;
if(lane == 0)
    __scoped_atomic_store_n(cluster_partials + wave, part,  // 部分和写进显存
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_CLUSTR);
__builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");        // 等写完成
__builtin_amdgcn_s_cluster_barrier();                       // cluster 同步
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");        // 作废 cache
const int row_groups = row_packs / kLamportWave;
const float v = lane < row_groups
                    ? __scoped_atomic_load_n(cluster_partials + row * row_groups + lane,
                                             __ATOMIC_RELAXED, __MEMORY_SCOPE_CLUSTR)
                    : 0.f;                                  // 从显存读回所有部分和
float total = __shfl(v, 0);
for(int g = 1; g < row_groups; ++g)
    total += __shfl(v, g);                                  // 串行加
```

### 3.5 现在的代码

host 端选模式（`新:2551`）：

```cpp
const bool split =
    row_packs > 128 && (world_size >= 4 ? rows <= 64 : packs <= 32768);
const int packs_per_thread = split ? 1 : row_block_packs(row_packs, rows);
const int row_threads      = row_packs / packs_per_thread;
// 发送块 256 个包；必须整除一行的包数，这样发送块不会越过本次调用的末尾
int send_packs = std::min(256, row_packs);
while(row_packs % send_packs != 0)
    send_packs -= kLamportWave;
const int senders = split ? (int)(packs / send_packs) : 0;
// 启动：<<<senders + rows, row_threads>>>
```

kernel 里的发送块（`新:1660`）：

```cpp
if constexpr(SPLIT)
{
    if((int)blockIdx.x < senders)
    {
        // 发送块和接收块一样大，只用前 send_packs 个线程，其余 wave 直接退出
        const int t =
            __builtin_amdgcn_readfirstlane(threadIdx.x / kLamportWave) * kLamportWave;
        if(t >= send_packs)
            return;
        const int p          = blockIdx.x * send_packs + threadIdx.x;   // 本线程负责的包
        const int wave       = __builtin_amdgcn_readfirstlane(p / kLamportWave);
        const uint32_t round = round_counter[wave];                      // tx 计数器
        pack v;
        v.data = reinterpret_cast<const P*>(input)[p];
        for(int j = 0; j < 4; ++j)
            v.i[j] = lamportCanonicalize(v.i[j]);                        // 0xffff → 0x7fff
        for(int g = 0; g < ngpus; ++g)
            if(g != rank)
                lamport_store_sys(lamport_slot(all[g], round, rank, ngpus, slot), p, v.i);
        // 回填 (round+2)%3：上一次调用用过的缓冲
        const uint32_t rearm = (round + 2) % kLamportBuffers;
        for(int g = 0; g < ngpus; ++g)
            if(g != rank)
                lamport_store_sys(lamport_slot(own, rearm, g, ngpus, slot), p, sentinel);
        if(threadIdx.x % kLamportWave == 0)
            round_counter[wave] = (round + 1) % kLamportBuffers;         // tx + 1
        return;
    }
}
// 下面是接收块（split）或 row 模式的 block
const int row = SPLIT ? (int)blockIdx.x - senders : (int)blockIdx.x;
...
round[k] = SPLIT ? rx_counter[wave] : round_counter[wave];             // 接收块用 rx
```

结尾更新计数器（`新:1897`）：

```cpp
if(threadIdx.x % kLamportWave == 0)
    for(int k = 0; k < PACKS; ++k)
    {
        const uint32_t next = (round[k] + 1) % kLamportBuffers;
        if constexpr(!SPLIT)
            round_counter[tid[k] / kLamportWave] = next;   // row 模式：tx 也由它写
        rx_counter[tid[k] / kLamportWave] = next;          // rx 总是写
    }
```

计数器分配加倍（`新:3558`）：`hipMalloc(&d_round, sizeof(uint32_t) * 2 * (slot / 32))`，前一半是 tx，后一半是 rx。

其他 Lamport kernel 结尾同时写两份（例如 AR+add，`新:2043`）：

```cpp
round_counter[wave]                       = (round + 1) % kLamportBuffers;  // tx
round_counter[slot / kLamportWave + wave] = (round + 1) % kLamportBuffers;  // rx
```

### 3.6 执行过程（6144×1，TP4）

```
host：row_packs = 768，split = true（768 > 128 且 1 行 ≤ 64）
      send_packs = 256（768 能被 256 整除），senders = 768 / 256 = 3
      启动 4 个 block，每个 768 线程

block 0、1、2（发送块，各在一个 CU 上）：
  wave 0~7（t < 256）：
    每个线程 p = blockIdx×256 + threadIdx，负责第 p 个包
    读 tx 计数器 → 读 input[p] → 把 0xffff 改成 0x7fff
    → 写到卡 1、2、3 的收件箱 → 把自己收件箱 (r+2)%3 号缓冲的 3 格写回 0xffff
    → lane 0 把 tx 计数器加 1 → 结束
  wave 8~23：一开始就退出

block 3（接收块，第 0 行）：
  768 个线程，线程 t 负责第 t 个包
  读 rx 计数器 → 读 input、residual、weight → 轮询 3 张卡
  → 求和、规约、写输出 → lane 0 把 rx 计数器加 1
```

2304 次跨卡写从 1 个 CU 变成分到 3 个 CU 上同时发；接收块的访存通道只用来轮询，尾部也没有回填。

M=64、hidden 6144 时：packs = 49152，senders = 192，加上 64 个接收块共 256 个 block，正好铺满 256 个 CU。

### 3.7 指令拆解

**新版：发送块的入口**（`new_bf16_split`，节选）

```
s_cmp_ge_i32 s51, s17            ; s51 = blockIdx.x，s17 = senders（预载的参数）；SCC = (blockIdx >= senders)
s_cbranch_scc0 733               ; 按 SCC 分成发送块、接收块两条路径（标量分支，整个 wave 一起走）
...
v_readfirstlane_b32 s0, v0       ; 取第一个活跃 lane 的 threadIdx
s_and_b32 s0, s0, 0x3e0          ; 低 5 位清零 = (threadIdx / 32) * 32，即 t
s_cmp_ge_i32 s0, s31             ; t >= send_packs ?（s31 = send_packs，预载的）
s_cbranch_scc1 170               ; 是 → 整个 wave 跳到结尾退出
```

| 指令 / builtin | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `__builtin_amdgcn_readfirstlane(x)` → `v_readfirstlane_b32 sD, vS` | 把第一个活跃 lane 的 VGPR 值复制到一个 SGPR | 把 `threadIdx.x / 32 * 32` 变成标量 | 编译器知道它对整个 wave 相同，`if(t >= send_packs) return` 就是整个 wave 一起跳转，不会出现半个 wave 退出的分歧 |
| `s_cmp_ge_i32` + `s_cbranch_scc1` | 标量比较，结果写 SCC 位；SCC 为 1 就跳转 | 判断这个 wave 是否闲置 | 标量分支只花几个时钟，闲置 wave 立刻结束，不占 CU |
| `p / kLamportWave` 再 `readfirstlane` | 同上 | 组号 wave 变成标量 | 读计数器 `round_counter[wave]` 时每个 wave 只读一次 |

**旧版：cluster 规约**（`old_bf16_cluster`，节选）

```
global_store_b32 v15, v14, s[2:3] scope:SCOPE_SE  ; 部分和写进显存
s_wait_storecnt_dscnt 0x0        ; 等所有写确认（包括前面的跨卡写）
s_barrier_signal_isfirst -1      ; block 内同步
s_barrier_wait 0xffff
s_barrier_signal -3              ; cluster 同步（-3 是 cluster 级 barrier 的编号）
s_barrier_wait 0xfffd
global_inv scope:SCOPE_SE        ; 作废本地 cache，之后才能读到别的 block 的写
global_load_b32 v14, v[0:1], off scope:SCOPE_SE   ; 读回部分和
s_wait_loadcnt 0x0
ds_bpermute_b32 v0, v15, v14     ; __shfl(v, 0)
...
ds_bpermute_b32 v1, v1, v14      ; 循环里每次 __shfl(v, g)，一次一个，串行
s_wait_dscnt 0x0
v_add_f32_e32 v0, v0, v1
```

| 指令 | 意思 | 这里的代价 |
|---|---|---|
| `global_store_b32 ... scope:SCOPE_SE` | 写显存，scope 表示这次写要对哪个范围可见（SE = 同一个 shader engine，cluster 内的 block 都在里面） | 部分和本来可以放 LDS，现在要走显存 |
| `s_wait_storecnt_dscnt 0x0` | 等到本 wave 所有未完成的写（storecnt）和 LDS 操作（dscnt）都归零 | 连前面 2304 次跨卡写的确认也一起等，多等一个跨卡来回（第 4 节详细讲） |
| `s_barrier_signal -3` / `s_barrier_wait 0xfffd` | gfx1250 的 cluster 级 barrier：发信号、等所有 block 到齐 | 跨 CU 的同步，比 block 内同步慢 |
| `global_inv scope:SCOPE_SE` | 作废这个范围内的本地 cache | 保证接下来读到的是别的 block 新写的值 |
| `ds_bpermute_b32` 循环 | 从指定 lane 取值，走 LDS 的交换电路 | 一行 24 组就是 24 次串行的取值 + 等待 |

### 3.8 通用性和经验

- **可推广的判断方法**：先算"每个 CU 要发出多少次访存"。如果一个 CU 上要排队发几千次远端写，而别的 CU 闲着，就该把发送拆出去。
- **拆分原则**：不需要协作的工作（逐包发送）可以任意分散；需要协作的工作（整行求和）尽量留在一个 block 里，用 LDS 完成。cluster 失败的原因就是把需要协作的部分也拆开了。
- **拆分之后检查共享状态**：原来同一个线程内的"读-改-写"（轮次计数器）一旦分给不同 block，就要考虑竞争。这里的解法是拆成两份各管各的，比加同步便宜。
- **"谁来做收尾工作"可以重新安排**：回填原来在接收方尾部，挪到发送块后，既离开了关键路径，也分散了。安排之前要用协议推一遍：这块缓冲此刻一定没人用吗？
- **适用条件**：小 M（block 数远少于 CU 数）时收益最大；block 数接近或超过 CU 数时就没有优势了，所以必须配合按形状选模式（第 12 节）。

---

## 4. Tip 2：block 同步只等 LDS，不等跨卡写的确认

### 4.1 提升多少

- POC 阶段测：2048×1 从 7.05 降到约 5.65 us。同一时期 peer 指针也改成了标量参数（约 0.1~0.4 us，第 6 节），扣掉以后这一项约 1 us。
- 消融测量：旧版规约（第 ⑤ 步）的总成本是 0.77 us（2048，row）到 1.76 us（6144，cluster），远超计算本身需要的时间。这个 tip 去掉的就是其中"等确认"的部分。
- **每次调用都至少有一次 block 同步，所以覆盖所有形状。**

### 4.2 原来哪里慢、怎么发现的

**先了解两个计数器。** GPU 执行写指令时，只是把写请求交出去，线程立刻往下走，不等写真正完成。每个 wave 有几个计数器记录"还有多少请求没完成"：

| 计数器 | 什么时候 +1 | 什么时候 −1 |
|---|---|---|
| `storecnt` | 发出一次显存写（包括跨卡写） | 收到"写好了"的确认 |
| `dscnt` | 发出一次 LDS 操作 | LDS 操作完成 |
| `loadcnt` | 发出一次显存读 | 数据返回 |

`s_wait_storecnt 0` 的意思是：停下来，直到之前所有写的确认都回来。

**`__syncthreads()` 编译出来是什么**（旧版汇编）：

```
s_wait_storecnt_dscnt 0x0     ; 等所有显存写的确认 + 所有 LDS 操作
s_barrier_signal -1           ; 告诉 block 里其他 wave：我到了
s_barrier_wait 0xffff         ; 等所有 wave 都到
```

`__syncthreads()` 的语义是："barrier 之前我写的任何内存，barrier 之后 block 里其他线程都能看到"。编译器不知道后面会读哪块内存，只能保守地把所有写都等完。

**问题在于跨卡写的确认来得很晚。** 设单程延迟为 L，4 张卡几乎同时开始发送：

```
时间 0   ：我写给对方，对方也写给我
时间 ≈ L ：对方的数据到了我的收件箱 → 我的轮询成功，开始算平方和
           同时，我的数据也到了对方那里，但"写好了"的确认才刚开始往回走
时间 ≈ 2L：我那几次跨卡写的确认才回到我这里
```

所以轮询刚成功时，我自己的跨卡写还差一趟回程。`__syncthreads()` 前的 `s_wait_storecnt` 会让每个 wave 在这里多等约一个 L。

**怎么发现的**：

1. 消融：跳过第 ⑤ 步规约，总时间少 0.77 us（2048，row）到 1.76 us（6144，cluster）。规约只是几十次加法和两次 LDS 读写，不应该这么慢。
2. 看汇编：在 barrier 前面看到了 `s_wait_storecnt_dscnt 0x0`。再看前面有哪些写还没确认，就是那 3 次跨卡写。
3. 对照：AR+add kernel 发完跨卡写直接开始轮询，汇编里中间没有 `s_wait_storecnt`。说明普通的写后面编译器不会插等待，只有 release 语义的同步才会插。

### 4.3 怎么改、为什么这样改

规约这一步，线程之间交换的只有 LDS 里的部分和 `partial[]`。barrier 之后也没有线程去读别人刚写的显存。所以只需要保证 LDS 的写对别人可见，不需要等显存写。

`__builtin_amdgcn_fence` 的第三个参数可以指定"只对哪类内存生效"，`"local"` 就是只管 LDS：

```cpp
fence(release, "workgroup", "local")  // 只要求我之前的 LDS 写对 block 内可见 → 只等 dscnt
s_barrier                              // 大家在这里碰头
fence(acquire, "workgroup", "local")  // 只要求之后能读到别人的 LDS 写
```

跨卡写的确认并没有消失，只是挪到了 kernel 结束时才等（GPU 报告 kernel 完成前必须等所有写落地）。那时它和后面的计算、写输出是重叠的，不在关键路径上。

**正确性**：Lamport 协议判断数据是否到达，靠的是收件箱里有没有 0xffff，不靠这个同步。所以去掉这个等待不影响协议。

### 4.4 原来的代码

`block_allreduce_f32`（`旧:816`，被 `ar_rmsnorm_gfx1250` 和旧版 Lamport kernel 共用）：

```cpp
template <bool kMax, bool parallel_waves = false>
DINLINE float block_allreduce_f32(float v)
{
    __shared__ float partial[32];
    __shared__ float result;
    for(int off = warpSize / 2; off > 0; off >>= 1)
    {
        const float o = __shfl_xor(v, off);
        v             = kMax ? fmaxf(v, o) : v + o;
    }
    if(threadIdx.x % warpSize == 0)
        partial[threadIdx.x / warpSize] = v;
    __syncthreads();                                    // ← 第 1 次，等跨卡写确认
    ...
    else if(threadIdx.x == 0)
    {
        float r = partial[0];
        for(int w = 1; w < (int)(blockDim.x / warpSize); ++w)
            r = kMax ? fmaxf(r, partial[w]) : r + partial[w];
        result = r;
    }
    __syncthreads();                                    // ← 第 2 次
    return result;
}
```

### 4.5 现在的代码

`新:1497`：

```cpp
// A block barrier that orders LDS only: __syncthreads() also waits for every
// outstanding global store of the wave, here the remote sends, whose acks take
// about one more fabric trip than the peers' data took to arrive.
DINLINE void lamport_lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
```

在 `lamport_row_sum`（`新:1586`）和 FP8 的 amax（`新:1859`）里使用。新版 Lamport kernel 里已经没有 `__syncthreads()`。

### 4.6 执行过程

```
轮询成功（对方数据到了），此时我的 3 次跨卡写还在等确认，storecnt = 3
每个 wave：wave 内求和 → lane 0 把部分和写进 LDS 的 partial[w]
旧：s_wait_storecnt_dscnt 0 → 等 LDS 写（几十个时钟）+ 等 3 次跨卡写的确认（约 1 us）→ barrier
新：s_wait_dscnt 0          → 只等 LDS 写（几十个时钟）→ barrier
之后：各线程读 partial[]，算 inv_rms，写输出 ...
kernel 结束前：硬件等 storecnt 归零，这时确认大概率早就回来了
```

### 4.7 指令拆解

旧版：

```
ds_store_b32 v15, v1              ; lane 0 把部分和写进 LDS
s_wait_storecnt_dscnt 0x0         ; 等 storecnt 和 dscnt 都归零
s_barrier_signal -1
s_barrier_wait 0xffff
```

新版：

```
ds_store_b32 v0, v14              ; 同上
s_wait_dscnt 0x0                  ; 只等 dscnt
s_barrier_signal -1
s_barrier_wait 0xffff
```

| builtin / 指令 | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `__builtin_amdgcn_fence(order, scope, "local")` | 内存栅栏。order：release（之前的写要先可见）或 acquire（之后的读要看到别人的写）；scope："workgroup" 表示 block 内；第三个参数限定只对 LDS（local）生效 | release 时只生成 `s_wait_dscnt 0`；acquire 在 LDS 上不需要额外指令 | 让编译器只等 LDS，不碰 storecnt |
| `__builtin_amdgcn_s_barrier()` → `s_barrier_signal -1` + `s_barrier_wait 0xffff` | gfx1250 的 barrier 拆成两条：signal 表示"我到了"，wait 等所有 wave 都 signal。`-1` 是 block 级 barrier 的编号 | block 内所有 wave 碰头 | 本身不带任何内存等待，等待由前后的 fence 决定 |
| `s_wait_storecnt_dscnt 0x0` | 同时等 storecnt 和 dscnt 归零 | 旧版 `__syncthreads` 生成的 | 被删掉的就是它的 storecnt 部分 |
| `s_wait_dscnt 0x0` | 只等 LDS 操作完成 | 新版生成的 | 几十个时钟 |

### 4.8 通用性和经验

- **适用条件，两条同时满足才能改**：
  1. barrier 前后交换的数据只在 LDS 里；
  2. barrier 之后没有线程读同 block 其他线程刚写的显存。
  
  如果 block 内要通过显存交换数据，必须保留完整的 `__syncthreads()`。
- **典型场景**：通信 kernel（发完远端写后马上做 block 规约）、写完大量输出后再做 block 规约的 kernel。写越多、写的目标越远，收益越大。
- **老架构上同样存在**：gfx942（MI300）上 `__syncthreads` 编译成 `s_waitcnt vmcnt(0) lgkmcnt(0)`，vmcnt 同时管读和写，也会等远端写的确认。
- **经验**：`__syncthreads()` 是"最保守的同步"，它的代价取决于之前发出了什么访存，而不是 barrier 本身。看到 barrier 慢，先查它前面生成了哪些 `s_wait_*`。
- cluster 路径改不了：它的部分和必须经过显存交换，必须用 cluster 级的 release，等待去不掉。这也是删掉 cluster 的原因之一。

---

## 5. Tip 3：一个线程负责多个包（PACKS）

### 5.1 提升多少

最终版代码，同一次测量（TP4，bf16，us）。PACKS=1 就是旧版的排法（一个线程一个包）：

| hidden×M | PACKS=1 | PACKS=2 | PACKS=4 | 最好的比 PACKS=1 |
|---|---|---|---|---|
| 4096×128 | 10.12 | **8.89** | 9.71 | −12% |
| 4096×256 | 13.19 | **11.90** | 12.61 | −10% |
| 4096×512 | 19.95 | 20.37 | **18.29** | −8% |
| 6144×128 | 14.06 | 13.33 | **11.97** | −15% |
| 6144×256 | 18.11 | 16.96 | **15.88** | −12% |
| 8192×128 | 15.96 | 14.82 | **13.21** | −17% |
| 8192×256 | 21.59 | 19.80 | **18.56** | −14% |

另外，旧版在 M=96~192 有一个台阶（6144：M=64 是 9.89，M=96 突然到 13.26）。原因是 cluster 模式的 block 总数超过了 256 个 CU。PACKS 让 row 模式的 block 变小、变少，这个台阶就没了。

### 5.2 原来哪里慢、怎么发现的

旧版一个线程只负责一个包，所以 block 大小 = 一行的包数 = hidden / 8：

| hidden | block 大小 | wave 数 |
|---|---|---|
| 2048 | 256 | 8 |
| 4096 | 512 | 16 |
| 8192 | 1024 | 32 |

hidden 8192 时一个 block 有 32 个 wave。M ≥ 128 时，行数已经接近或超过 CU 数，再大的 block 也不能让更多 CU 干活，反而带来这些问题：

- block 同步要等 32 个 wave 全部到齐，最慢的那个 wave 决定所有人什么时候继续。
- 大 block 占的资源多，一个 CU 能同时放的 block 少。M 大到 block 数超过 CU 数时，后面的行要等前面的 block 结束才能上；小 block 可以在一个 CU 上同时放几个，一个在轮询等数据时，另一个可以计算。
- 每个线程只有一个包的工作，发出一次读就得等，没有别的独立工作能填补等待。

**怎么发现的**：扫参数。同一个形状跑 PACKS=1/2/4/8 各一遍，再看 M 从 1 到 2048 的变化趋势（第 12 节的选择规则就是从这张表来的）。上面的机制是我的解释，没有用计数器逐条验证，结论以测量为准。

### 5.3 怎么改、为什么这样改

线程 t 负责 t、t+B、t+2B …… 共 PACKS 个包（B = block 的线程数）：

```
hidden 8192，PACKS=4，block = 1024 / 4 = 256 个线程
线程 0   负责包 0、256、512、768
线程 1   负责包 1、257、513、769
...
线程 255 负责包 255、511、767、1023
```

**为什么是间隔 B，而不是连续的 4 个包**：

- 访存合并：同一时刻，相邻线程访问相邻的包，一个 wave 的 32 次读写落在连续的 512 字节里。
- 规约顺序：car_fused 的规约是"wave 内蝴蝶求和，再按 wave 编号顺序串行加"。按间隔 B 排，第 k 份包落在原来的第 `w + k × (B/32)` 号 wave 上，把每份的部分和放到 `partial[w + k × waves]`，串行加的顺序就和 car_fused 完全一样，结果逐位一致。

**选几个包**：取能让 block ≤ 256 线程的最小 PACKS；行数多于 256 时限 128 线程，多于 512 时限 64 线程（第 12 节）。

**配套修改**：代码形状变了以后，编译器把一部分"先乘再加"合成了一条乘加指令，结果差一个最低位，所以加了 `#pragma clang fp contract(off)`（第 13 节）。

### 5.4 原来的代码

`旧:1539`：

```cpp
const int row_packs = hidden / pack_size;
const int tid       = blockIdx.x * blockDim.x + threadIdx.x;   // 一个线程一个包
const int row       = tid / row_packs;
const int col       = tid - row * row_packs;
...
pack local;
local.data = reinterpret_cast<const P*>(input)[tid];
```

### 5.5 现在的代码

模板参数 `PACKS`，每个包的数据都是长度为 PACKS 的数组（`新:1697`）：

```cpp
const int block = row_packs / PACKS;          // = blockDim.x，但不用读 blockDim（第 6 节）
const int row   = SPLIT ? (int)blockIdx.x - senders : (int)blockIdx.x;
const int tid0  = row * row_packs + threadIdx.x;
int tid[PACKS];
uint32_t round[PACKS];
for(int k = 0; k < PACKS; ++k)
{
    tid[k] = tid0 + k * block;                // 线程 t 负责 t, t+B, t+2B ...
    const int wave = __builtin_amdgcn_readfirstlane(tid[k] / kLamportWave);
    round[k]       = SPLIT ? rx_counter[wave] : round_counter[wave];
}
```

规约里部分和的位置（`新:1575`）：

```cpp
template <int PACKS>
DINLINE float lamport_row_sum(const float (&v)[PACKS], int row_groups, int waves)
{
    __shared__ float partial[32];
    const int lane = threadIdx.x % kLamportWave;
    for(int k = 0; k < PACKS; ++k)
    {
        const float p = lamport_wave_sum(v[k]);
        if(lane == 0)
            partial[threadIdx.x / kLamportWave + k * waves] = p;   // 第 k 份放在 w + k*waves
    }
    lamport_lds_barrier();
    float r = partial[0];
    for(int g = 1; g < row_groups; ++g)                            // 按组号顺序串行加
        r += partial[g];
    return r;
}
```

host 端为每种 PACKS 实例化一个 kernel（`新:2568`），`MAX_THREADS = 1024 / PACKS` 向上取到 32 的倍数，作为 `__launch_bounds__`：

```cpp
lamport_ar_rmsnorm_gfx1250<T, OutT, NG, (kLamportRowMaxThreads / PK + 31) / 32 * 32, PK, SPLIT>
    <<<senders + (int)rows, row_threads, 0, stream>>>(...)
```

### 5.6 执行过程（8192×128，TP4，PACKS=4）

```
host：row_packs = 1024，split = false（128 行 > 64）
      row_block_packs：行数 ≤ 256 → block 最多 256 线程；1024/1=1024、1024/2=512 都超，1024/4=256 → PACKS=4
      启动 128 个 block，每个 256 线程（8 个 wave）

每个 block（一行）：
  线程 t 读 4 个计数器、4 个 input 包 → 4×3 次跨卡写 → 等一次参数读 → 读 4 个 residual、4 个 weight
  → 轮询 4×3 个格子 → 4 个包各自求和、加 residual、平方和
  → 4 次 wave 内求和，lane 0 写 partial[w]、partial[w+8]、partial[w+16]、partial[w+24]
  → 一次同步 → 每个线程按 0..31 顺序加 32 个部分和 → inv_rms
  → 写 4 个输出包 → 4×3 次回填
```

### 5.7 指令拆解

| builtin / 写法 | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `__launch_bounds__(MAX_THREADS, 1)` | 告诉编译器 block 最多多少线程 | PACKS=4 时上限 256 | 线程少，每个线程可分到更多 VGPR，存得下 4 个包的数据，不溢出到 scratch |
| `#pragma unroll` + 编译期常量 PACKS | 循环完全展开 | `tid[k]`、`local[k]` 等数组都变成独立的寄存器 | 数组只用常量下标访问才能留在寄存器里，用运行时下标会被放进 scratch 内存 |
| `__builtin_amdgcn_readfirstlane(tid[k] / 32)` | 见第 3.7 节 | 每个包的组号变成标量 | 读 `round_counter[wave]` 用一次标量读 |

### 5.8 通用性和经验

- **"一行一个 block，一个线程一个元素"不一定最好。** hidden 大时 block 会很大。block 大小应该独立于问题规模来选，用"一个线程多个元素"去适配。
- **按间隔排而不是连续排**：既保证同一时刻访存合并，又能让规约顺序和原来一致。需要逐位一致时，先想清楚新的数据排布对应原来哪个顺序。
- **PACKS 用模板参数**，让循环完全展开、数组留在寄存器里。代价是要编译多个实例，host 端按形状选。
- **改了代码形状要重新做逐位比对**：编译器的浮点合并、指令调度会跟着变（第 13 节就是这么发现的）。

---

## 6. Tip 4：kernel 参数全部放进预载区

### 6.1 提升多少

- 开发中期（11-24）在 split 模式测，6144×1：bf16 从 6.59 降到 6.10，FP8 从 7.08 降到 6.43。这一步做了参数重排、gemma_norm 并进 rank、去掉 gridDim / blockDim 的读取。
- 更早的 POC 阶段，把 peer 指针从结构体改成 8 个标量参数，约 0.1~0.4 us。
- 对 FP8 特别重要：FP8 输出要用的 `scale_out`、`bf16_output` 原来在轮询之后才读，正好卡在关键路径上。

### 6.2 原来哪里慢、怎么发现的

**kernel 参数是怎么到 kernel 里的。** host 启动 kernel 时，把参数打包成一块内存（kernarg 段）。kernel 启动后只拿到这块内存的地址，放在 `s[0:1]` 两个 SGPR 里。用到哪个参数，就发一条标量读 `s_load` 去取，再用 `s_wait_kmcnt` 等它回来，这一来一回要几百个时钟。

**预载（preload）**：aiter 编译时加了 `-mllvm --amdgpu-kernarg-preload-count=32`（`aiter/jit/core.py:1546`）。wave 启动前，硬件直接把参数区开头的若干个 dword 放进 SGPR，kernel 不用再读。**实测只有前 30 个 dword（偏移 0~119 字节）被预载**：看汇编，偏移 120 以后的参数都有 `s_load`，之前的都没有。这和"启动时最多 32 个用户 SGPR，其中 2 个被参数区地址 `s[0:1]` 占用"的解释一致。

**旧版的参数布局**（字节偏移）：

| 偏移 | 参数 | 在预载区？ | 实际怎么读 |
|---|---|---|---|
| 0~39 | input, residual_inp, residual_out, output, weight | 是 | 在 SGPR 里 |
| 48~111 | `LamportPeers peers`（8 个指针的结构体） | 在范围内，但用不上 | 代码用运行时下标 `peers.ptrs[(rank+i)%ngpus]` 取，寄存器不能按运行时下标取，只能每次从内存读 |
| 112 | cluster_partials | — | row 模式不用 |
| 120 | round_counter | **否** | `s_load` |
| 128~151 | slot, rank, hidden, eps, gemma_norm | **否** | `s_load` |
| 152~167 | scale_out, bf16_output | **否** | `s_load`，FP8 时在轮询之后等 |
| 168~ | 隐藏参数（gridDim、blockDim 等） | **否** | 代码用了 `blockDim.x`，要读两次 |

结构体占了 64 字节，把后面发送要用的 round_counter、slot、rank 都挤出了预载区。

**旧版汇编里看到的**（`old_bf16_row`，节选）：

```
s_load_b32 s12, s[0:1], 0xa8          ; 读隐藏参数：x 方向的 block 数
s_wait_kmcnt 0x0                       ; 等它
...                                    ; 根据 blockIdx 算 blockDim 存放的位置
global_load_u16 v1, v1, s[12:13]       ; 再读一次：blockDim.x
s_wait_loadcnt 0x0                     ; 等它 —— 这时还没开始读 input
v_readfirstlane_b32 s12, v1
v_mad_u32 v32, s14, s27, v0            ; tid = blockIdx * blockDim + threadIdx
global_load_b128 v[2:5], v32, s[2:3]   ; 现在才开始读 input
s_load_b128 s[12:15], s[0:1], 0x80     ; 读 slot, rank, hidden, eps
s_load_b64 s[16:17], s[0:1], 0x78      ; 读 round_counter
s_wait_kmcnt 0x0
...
s_load_b32 s26, s[16:17], 0x0          ; 读 round_counter[wave]
s_load_b64 s[22:23], s[22:23], 0x0     ; 读 peers.ptrs[(rank+1)%4]
s_wait_kmcnt 0x0
global_store_b128 v32, v[10:13], s[22:23] scope:SCOPE_SYS   ; 第 1 次跨卡写
s_load_b64 s[28:29], s[28:29], 0x0     ; 读 peers.ptrs[(rank+2)%4]
s_wait_kmcnt 0x0
global_store_b128 v32, v[10:13], s[28:29] scope:SCOPE_SYS   ; 第 2 次
s_load_b64 s[28:29], s[34:35], 0x0     ; 读 peers.ptrs[(rank+3)%4]
s_wait_kmcnt 0x0
global_store_b128 v32, v[10:13], s[28:29] scope:SCOPE_SYS   ; 第 3 次
```

从 kernel 开始到第一次跨卡写，中间串行等了 4~5 次内存读；每次跨卡写前面还要等它的目标地址读回来。FP8 版本还有：

```
s_load_b128 s[4:7], s[0:1], 0x98       ; 读 scale_out, bf16_output（轮询之后才发出）
...
s_wait_kmcnt 0x0                       ; 在 epilogue 里等它
```

**怎么发现的**：看汇编时专门找两类指令：

1. `s_load ... s[0:1], 偏移`：凡是从 `s[0:1]` 读的，都是没预载的参数。
2. `s_wait_kmcnt`：看它出现在关键路径的什么位置（跨卡写前、轮询后）。

再把一个参数挪到不同偏移，看汇编里它有没有 `s_load`，就测出了 120 字节这条边界。

### 6.3 怎么改、为什么这样改

目标：**发送和 epilogue 用到的每个参数都在前 120 字节里，kernel 里不出现任何对 `s[0:1]` 的读**（TP ≤ 4 时）。

1. **peer 指针拆成 8 个标量参数 p0~p7**，用编译期常量下标选择：

   ```cpp
   void* const all[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
   for(int g = 0; g < ngpus; ++g)          // 展开后 g 是常量
       if(g != rank)
           lamport_store_sys(lamport_slot(all[g], ...), ...);
   ```

   展开后每次只用 `all[0]`、`all[1]`……，编译器直接用对应的 SGPR。`if(g != rank)` 是标量比较加跳转。自己的收件箱 `own` 也用一串比较选出来（`s_cselect_b32`），不读内存。

2. **按使用顺序重排参数**，TP4 用不到的 p4~p7 放到最后。

3. **gemma_norm 并进 rank 的第 8 位**：前 120 字节已经放满，没有位置再放一个参数。rank 最大是 7，只用低 8 位里的 3 位。

4. **不读 gridDim、blockDim**：它们是隐藏参数，在参数区的末尾，不在预载区。
   - blockDim：`block = row_packs / PACKS`，row_packs 由预载的 hidden 算出。
   - gridDim：发送块原来要用它判断越界。现在让 `send_packs` 整除一行的包数，发送块永远不会越过本次调用的末尾，不需要判断。

5. **删掉 cluster_partials**（cluster 已删）。

**新版的参数布局**：

| 偏移 | 参数 | SGPR |
|---|---|---|
| 0~39 | input, residual_inp, residual_out, output, weight | s[2:3] ~ s[10:11] |
| 40 | round_counter | s[12:13] |
| 48 | slot | s14 |
| 52 | rank_gemma（rank + gemma_norm 在第 8 位） | s15 |
| 56 | hidden | s16 |
| 60 | senders | s17 |
| 64~95 | p0, p1, p2, p3 | s[18:19] ~ s[24:25] |
| 96 | scale_out | s[26:27] |
| 104 | bf16_output | s[28:29] |
| 112 | eps | s30 |
| 116 | send_packs | s31 |
| 120~ | p4 ~ p7（不在预载区，TP8 才用） | — |

### 6.4 原来的代码

`旧:1510`：

```cpp
template <typename T, typename OutT, int ngpus, int MAX_THREADS, bool CLUSTER>
__global__ void __launch_bounds__(MAX_THREADS, 1) lamport_ar_rmsnorm_gfx1250(
    const T* __restrict__ input,
    const T* __restrict__ residual_inp,
    T* __restrict__ residual_out,
    OutT* __restrict__ output,
    const T* __restrict__ weight,
    LamportPeers peers,                   // 8 个指针的结构体
    float* __restrict__ cluster_partials,
    uint32_t* __restrict__ round_counter,
    int slot,
    int rank,
    int hidden,
    float eps,
    bool gemma_norm,
    float* __restrict__ scale_out,
    T* __restrict__ bf16_output)
{
    ...
    const int tid       = blockIdx.x * blockDim.x + threadIdx.x;   // 读 blockDim
    ...
    for(int i = 1; i < ngpus; ++i)
    {
        P* dst = reinterpret_cast<P*>(peers.ptrs[(rank + i) % ngpus]) +   // 运行时下标
                 (round * ngpus + rank) * slot + tid;
        lamport_store_b128(dst, local.i);
    }
```

### 6.5 现在的代码

`新:1607`：

```cpp
template <typename T, typename OutT, int ngpus, int MAX_THREADS, int PACKS, bool SPLIT>
__global__ void __launch_bounds__(MAX_THREADS, 1) lamport_ar_rmsnorm_gfx1250(
    const T* __restrict__ input,
    const T* __restrict__ residual_inp,
    T* __restrict__ residual_out,
    OutT* __restrict__ output,
    const T* __restrict__ weight,
    uint32_t* __restrict__ round_counter, // [2][slot / 32] tx, rx
    int slot,
    int rank_gemma,                       // rank, and gemma_norm in bit 8
    int hidden,
    int senders,                          // SPLIT only
    void* p0, void* p1, void* p2, void* p3,
    float* __restrict__ scale_out,        // FP8 only
    T* __restrict__ bf16_output,          // FP8 only, may be null
    float eps,
    int send_packs,                       // SPLIT only, divides the packs of a row
    void* p4,                             // not preloaded from here on
    void* p5, void* p6, void* p7)
{
    ...
    const int rank        = rank_gemma & 0xff;
    const bool gemma_norm = (rank_gemma >> 8) & 1;

    // static indices only: a runtime index into the inboxes is a scalar load
    // in series before each remote store
    void* const all[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    char* own          = static_cast<char*>(all[0]);
    for(int g = 1; g < ngpus; ++g)
        own = g == rank ? static_cast<char*>(all[g]) : own;
    ...
    const int block = row_packs / PACKS; // = blockDim.x, without a kernel argument load
```

host 端传参（`新:2562`）：

```cpp
#define LAMPORT_RMS_ARGS                                                                    \
    input, residual_inp, residual_out, output, weight, d_round, slot,                       \
        rank | (gemma_norm ? 0x100 : 0), hidden, senders, peers.ptrs[0], peers.ptrs[1],     \
        peers.ptrs[2], peers.ptrs[3], scale_out, bf16_out, eps, send_packs, peers.ptrs[4],  \
        peers.ptrs[5], peers.ptrs[6], peers.ptrs[7]
```

### 6.6 执行过程

```
wave 启动前：硬件把参数区前 120 字节放进 s2 ~ s31
kernel 第 1 条有用指令起：
  用 s16（hidden）、blockIdx、threadIdx 算出 tid → 立刻读 input
  用 s15 & 0xff 得到 rank；用比较 + s_cselect 从 s[18:25] 里选出 own
  读 round_counter[wave]（唯一一次要等的读，它本来就在显存里）
  → 直接用 s[18:19]、s[20:21]... 作为基址发跨卡写，中间不再等任何参数
epilogue：eps 在 s30，gemma 在 s15 的第 8 位，scale_out 在 s[26:27]，都不用读
```

### 6.7 指令拆解

新版开头（`new_bf16_split`，节选）：

```
s_and_b32 s50, s15, 0xff          ; rank = rank_gemma & 0xff
s_cmp_eq_u32 s50, 1               ; rank == 1 ?
s_cselect_b32 s1, s20, s18        ; 是 → 取 p1 的低 32 位，否则 p0 的
s_cselect_b32 s15, s21, s19       ; 高 32 位同理
s_cmp_eq_u32 s50, 2
s_cselect_b32 s26, s22, s1        ; rank == 2 → p2，否则保留上一步的结果
...                               ; 最后得到 own，全程在 SGPR 里
```

新版 epilogue（`new_bf16_row1`，节选）：

```
s_bitcmp0_b32 s17, 8              ; 测 rank_gemma 的第 8 位（gemma_norm）
s_cselect_b32 s2, 0, 1.0          ; gemma 时 weight 加 1.0，否则加 0
v_add_f32_e32 v0, s30, v0         ; 平方和 / hidden + eps，eps 直接用 s30
```

| 指令 | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `s_load_b32/b64/b128 sD, s[0:1], 偏移` | 从参数区按偏移读 1/2/4 个 dword 到 SGPR | 旧版读没预载的参数 | 新版里不应该再出现（kernarg wait 那一条是故意的，第 7 节） |
| `s_wait_kmcnt 0x0` | 等所有标量读（km = scalar memory）返回 | 旧版每次跨卡写前都有 | 新版发送前不再需要 |
| `global_load_u16` 读隐藏参数 | 读 2 字节 | 旧版读 blockDim.x | 新版用 hidden / PACKS 代替 |
| `s_cmp_eq_u32` + `s_cselect_b32` | 标量比较，然后按 SCC 二选一 | 从 p0~p3 里选出 own | 不需要内存，几个时钟 |
| `s_bitcmp0_b32 sX, 8` | 测试 sX 的第 8 位是否为 0，结果进 SCC | 取 gemma_norm | 布尔量并进 rank，省一个参数位 |

### 6.8 通用性和经验

- **检查方法**：反汇编后搜 `s[0:1]`，每一个 `s_load ... s[0:1]` 都是一次没预载的参数读。再看它的 `s_wait_kmcnt` 落在哪里。
- **排参数的原则**：
  - 热参数（启动后马上用、或关键路径上用）放前面；
  - 不要用结构体 / 数组加运行时下标传指针，拆成标量，用展开后的常量下标选；
  - 低频或只在大 TP 下用的参数放后面；
  - 空间不够时，小整数和布尔可以并进同一个 int。
- **不要读 `blockDim`、`gridDim`**：它们是隐藏参数，放在参数区末尾，一定不在预载区。能用编译期常量或已有参数算出来就算出来。
- **预载上限**：这台机器、这个编译器上实测 30 个 dword。换架构或编译器后要重新看汇编确认。
- **代价**：参数一多就会超出预载区，所以每加一个参数都要看它落在哪个偏移。

---

## 7. Tip 5：row 模式发完后，等一次参数读

### 7.1 提升多少

移植版（所有参数都已预载）上，row 模式加不加这一步（TP4，bf16，us）：

| hidden×M | 不加 | 加 | 变化 |
|---|---|---|---|
| 2048×1 | 6.01 | 5.43 | −0.6 |
| 6144×1 | 10.40 | 7.52 | −2.9 |
| 8192×1 | 11.53 | 8.25 | −3.3 |
| 6144×64 | 12.62 | 10.14 | −2.5 |
| 8192×64 | 13.73 | 11.26 | −2.5 |

数字很大，但要注意两点：

1. **基线是"所有参数都预载了"的版本。** main 原版其实已经"碰巧"有一次类似的等待（见 7.2），所以相对 main，这一项是补回来，不是净收益。
2. **机制没有证实。** 这是实验得出的结论。

### 7.2 原来哪里慢、怎么发现的

**发现过程：一次性能回退的二分。**

1. 11-18 的 POC 版本（v7）在 row 模式下很快。移植到新 main 之后，row 模式 6144×1 从 7.6 变成 10.4 us。
2. 用 11-18 的会话记录，从起点开始逐步重放当天对 `.cuh` 的每一次修改，每步存一份快照。重放到最后和 POC 的提交逐字节一致，说明快照可靠。
3. 逐个编译、测试关键快照，找到变慢的那一步：一次大的重写，删掉了一个调试参数 `dbg`。
4. 对比这两步的汇编，唯一的差别是：v7 在发完跨卡写之后、开始轮询之前，读了一次 `dbg`（它是最后一个参数，不在预载区）并等它返回。
5. 在移植版里只补这一次读（不加参数，直接读参数区偏移 128），数字完全回到了 v7。

**main 原版也有这一步，是碰巧的**：旧版轮询时要用自己的收件箱地址 `peers.ptrs[rank]`，它也是从参数区读的（`旧:1568`）。汇编里，这次读正好落在最后一次跨卡写和第一次轮询之间：

```
global_store_b128 v32, v[10:13], s[28:29] scope:SCOPE_SYS   ; 最后一次跨卡写
...
s_load_b64 s[4:5], s[2:3], 0x0     ; 读 peers.ptrs[rank]（参数区里）
...
s_wait_kmcnt 0x0                   ; 等它
...
global_load_b128 v[26:29], v[34:35], off scope:SCOPE_SYS    ; 第一次轮询
```

所以：peer 指针全部预载（第 6 节）之后，这次等待也跟着没了，row 模式反而变慢。

**为什么有效：推测，没有证实。** 我的推测是，这次标量读要排在刚发出的跨卡写后面才能返回，相当于"等写都发出去了再开始轮询"。如果轮询太早开始，轮询的读请求会和还没发完的跨卡写抢同一个 CU 的访存通道，数据反而到得晚。这个等待的长短会随发送量自动变化。

**试过的替代写法，都不行**：

| 写法 | 结果 |
|---|---|
| 发完后 `s_sleep` 固定时长 | 大 hidden 最好只到约 8.1（6144），2048 没改善；睡太久（127）把 2048 拖到 7.6 |
| 轮询没等到就退避睡一下 | 没效果 |
| `s_wait_xcnt 0`（等写的数据离开寄存器） | 没效果 |
| 读一个设备显存地址（system scope 的标量读） | M 大时慢到 68~261 us，不能用 |
| 环境变量 `HIP_FORCE_DEV_KERNARG` 设 0 或 1 | 结果一样，说明和参数区放在哪块内存无关 |

**split 模式的接收块不加**：接收块不发送，加了只会变慢（测过）。

### 7.3 怎么改、为什么这样改

既然机制没查清，就用最直接、可复现的写法：在 row 模式发完之后，读一次参数区偏移 128（第一个不在预载区的 dword，读到的值不用），等它返回，再读 residual/weight、开始轮询。

- 偏移 128 的位置现在是 p5 的低 32 位，TP4 用不到，读它没有副作用。
- 用 `volatile` 防止编译器删掉这次读；用一条空的内联汇编"使用"这个值，强制编译器在这里等它返回。

### 7.4 原来的代码

main 里没有专门的代码，等待来自轮询地址的读（`旧:1568`）：

```cpp
const P* inbox = reinterpret_cast<const P*>(peers.ptrs[rank]) + round * ngpus * slot + tid;
```

### 7.5 现在的代码

`新:1509`：

```cpp
// Waits for a kernel argument past the preloaded ones (offset 128) to come
// from memory. A block that sends a row and then polls the inbox gets the
// peers' data sooner when its first poll waits for this load: TP4 bf16 row of
// 6144, one block a row, 10.4 -> 7.6 us. An s_sleep of any length does not
// give the same gain (the load seems to return after the sends left).
DINLINE void lamport_kernarg_wait()
{
    const __attribute__((address_space(4))) volatile int* args =
        (const __attribute__((address_space(4))) volatile int*)
            __builtin_amdgcn_kernarg_segment_ptr();
    const int v = args[128 / 4];
    asm volatile("" ::"s"(v));
}
```

调用处（`新:1721`），只在 row 模式：

```cpp
if constexpr(!SPLIT)
{
    for(int k = 0; k < PACKS; ++k)
        for(int g = 0; g < ngpus; ++g)
            if(g != rank)
                lamport_store_sys(lamport_slot(all[g], round[k], rank, ngpus, slot), tid[k], local[k].i);
    lamport_kernarg_wait();
}
// 之后读 residual、weight，开始轮询
```

### 7.6 执行过程

```
发出最后一次跨卡写（指令发出即返回，写还在路上）
→ s_load 读参数区偏移 128
→ s_wait_kmcnt 0：等这次读返回（推测：返回时刚才的写大致都已离开本 CU）
→ 读 residual、weight
→ 开始轮询
```

### 7.7 指令拆解

新版汇编（`new_bf16_row1`）：

```
buffer_store_b128 v[10:13], v1, s[44:47], null offen th:TH_STORE_BYPASS scope:SCOPE_SYS  ; 最后一次跨卡写
s_load_b32 s0, s[0:1], 0x80       ; 读参数区偏移 128
v_lshl_add_u64 v[14:15], v[26:27], 4, s[4:5]
s_wait_kmcnt 0x0                  ; 等它返回
global_load_b128 v[6:9], v[14:15], off        ; residual
global_load_b128 v[2:5], v0, s[10:11] scale_offset   ; weight
```

| builtin / 写法 | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `__builtin_amdgcn_kernarg_segment_ptr()` | 返回参数区的地址（就是 `s[0:1]`），地址空间 4（常量内存） | 拿到参数区起点 | 不用新增参数就能读偏移 128 |
| `volatile` 读 `args[128/4]` → `s_load_b32 s0, s[0:1], 0x80` | 标量读一个 dword | 读一个不在预载区的位置 | `volatile` 保证这次读不会被删掉或挪走 |
| `asm volatile("" ::"s"(v))` | 一条空汇编，声明它要用 v，并且 v 必须在 SGPR 里 | 迫使编译器在这里插入 `s_wait_kmcnt 0` | 把"等这次读返回"固定在这个位置 |
| `s_wait_kmcnt 0x0` | 等标量读返回 | 等待本身 | 推迟轮询开始的时间 |

### 7.8 通用性和经验

- **这个 tip 本身不建议推广**：机制不明，换硬件、换驱动可能失效。代码注释里写了测量依据，方便以后复查。
- **值得推广的是发现它的方法**：
  - 性能回退时，**二分**到具体的那一次修改，再**对比两版汇编**，比猜原因可靠得多。
  - 一次看起来无关的读（调试参数），可能通过改变时序影响性能。删"无用"代码之后也要重测。
- **启发**："轮询开始得太早"本身可能有害。通信 kernel 里，发送和轮询共用访存通道时，先让发送走完可能更快。split 模式把发送和轮询放到不同的 CU 上，从结构上解决了这个问题，所以 split 不需要这一步。

---

## 8. Tip 6：FP8 的 amax 按整数比较，用 LDS 原子 max 汇总

### 8.1 提升多少

- 消融测量：FP8 求整行最大值这一段的成本，约 0.45 us 降到约 0.2 us（开发后期，split 模式）。
- FP8 输出比 bf16 输出多花的时间（最终版对 main，TP4 中位数）：**0.89 → 0.25 us**，即从 bf16 的 1.09 倍降到 1.03 倍。这是本 tip、第 6 节（scale_out、bf16_output 进预载区）、第 9 节（DPP）、删 cluster 的合计。
- 做完这一步后，所有测过的 FP8 形状都不比 Gluon 慢（6144×1 最接近，0.99）。

### 8.2 原来哪里慢、怎么发现的

FP8 输出要在 RMSNorm 之后，再求一次整行 |out| 的最大值 amax。旧版用的是和平方和一样的通用规约 `block_allreduce_f32<true, true>`：

```
平方和：wave 内 5 次 ds_bpermute → 写 LDS → 同步① → 0 号线程串行加 → 写 LDS → 同步② → 读
amax  ：wave 内 5 次 ds_bpermute → 写 LDS → 同步③ → 第 0 个 wave 读所有部分值
        → 再做 5 次 ds_bpermute → 写 LDS → 同步④ → 读
```

旧版 FP8 汇编里一共 **4 次 block 同步、15 次 `ds_bpermute`**，其中求部分值的两次同步前还有 `s_wait_storecnt`（第 4 节的问题）。另外 `scale_out`、`bf16_output` 在轮询之后才从参数区读（第 6 节）。cluster 模式更慢：amax 要再走一次显存 + cluster 同步。

**怎么发现的**：

1. 消融：只跳过 amax 的规约，FP8 快了约 0.45 us。
2. 看汇编：数同步、`ds_bpermute`、`s_wait_kmcnt` 的个数和位置。

还有一个细节：浮点 max（`v_max_num_f32`）要先把输入"规范化"（处理 NaN），旧版汇编里每一步都多一条 `v_max_num_f32 v11, v11, v11`。

### 8.3 怎么改、为什么这样改

**关键观察：amax 不可能是负数，也不可能是 NaN。**

- 它从 0 开始，每次和 `fabsf(x)` 取 `fmaxf`，`fabsf` 结果非负。
- `fmaxf(a, NaN)` 返回 a，NaN 会被丢掉。全是 NaN 时 amax 保持 0。

**非负浮点数的大小顺序，和它们的二进制位当作无符号整数的大小顺序完全一致。** float 的 32 位是：

```
[符号 1 位][指数 8 位][尾数 23 位]
非负数符号位都是 0；指数大的数更大，指数相同时尾数大的数更大
→ 整个 32 位当无符号整数比较，结果和浮点比较一样
例：1.0 = 0x3f800000，2.0 = 0x40000000，0x40000000 > 0x3f800000
```

所以 amax 可以全程用整数 max 来算，好处：

1. 整数 max 不需要处理 NaN，少一条规范化指令；还能和 DPP 合成一条 `v_max_u32_dpp`（第 9 节）。
2. **跨 wave 可以用原子操作。** 原子 max 的结果和执行顺序无关（max 是精确运算，不会有舍入），所以不用像求和那样必须按固定顺序加。每个 wave 用一次 LDS 原子 max 把自己的结果合进一个字，就完成了跨 wave 的汇总，不需要部分值数组和第二轮规约。

再省一次同步：

3. **清零放在求和的同步之前。** 原子 max 需要先把那个字清零，而清零本身也需要同步才能让所有 wave 看到。做法是 0 号线程在平方和的同步之前就写 0，于是平方和那次同步顺便保证了清零完成。之后只要一次同步，等所有 wave 的原子 max 做完。

**为什么求和不能这样做**：浮点加法换顺序结果会差最低位，原子加的执行顺序不固定，就做不到和 car_fused 逐位一致。max 没有这个问题。

改完后 FP8 只有 **2 次同步（平方和 1 次 + amax 1 次）、0 次 `ds_bpermute`、0 次参数读**。

### 8.4 原来的代码

`旧:1668`（非 cluster 路径）：

```cpp
float amax = 0.f;
for(int j = 0; j < pack_size; ++j)
    amax = fmaxf(amax, fabsf(upcast_s(o[j])));
if(bf16_output != nullptr)
    reinterpret_cast<P*>(bf16_output)[tid] = o;
...
amax = block_allreduce_f32<true, true>(amax);          // 两次 __syncthreads，10 次 ds_bpermute
const float fp8_max = opus::cast<opus::fp32_t>(opus::numeric_limits<OutT>::max());
const float scale   = amax * (float)(1. / fp8_max);
const float inv     = __builtin_amdgcn_rcpf(scale);
```

`block_allreduce_f32` 的 `parallel_waves` 分支（`旧:831`）：

```cpp
if(threadIdx.x < warpSize)
{
    float r = threadIdx.x < blockDim.x / warpSize ? partial[threadIdx.x] : 0.f;
    for(int off = warpSize / 2; off > 0; off >>= 1)
    {
        const float o = __shfl_xor(r, off);
        r             = kMax ? fmaxf(r, o) : r + o;
    }
    if(threadIdx.x == 0)
        result = r;
}
...
__syncthreads();
return result;
```

### 8.5 现在的代码

清零，放在平方和规约之前（`新:1809`）：

```cpp
// FP8: the row amax, merged by an LDS atomic max after the sum barrier
__shared__ uint32_t amax_bits;
if constexpr(QUANT)
    if(threadIdx.x == 0)
        amax_bits = 0;
const int row_groups = row_packs / kLamportWave;
const float inv_rms =
    rsqrtf(lamport_row_sum<PACKS>(square_sum, row_groups, block / kLamportWave) / hidden + eps);
    //     ↑ 里面的同步顺便让所有 wave 看到 amax_bits = 0
```

求 amax（`新:1842`）：

```cpp
uint32_t a = 0;
for(int k = 0; k < PACKS; ++k)
{
    float amax = 0.f;
    for(int j = 0; j < pack_size; ++j)
        amax = fmaxf(amax, fabsf(upcast_s(o[k][j])));   // 和 car_fused 相同的逐元素步骤
    const uint32_t b = __builtin_bit_cast(uint32_t, amax);
    a                = a > b ? a : b;                     // 本线程几个包之间：整数 max
    if(bf16_output != nullptr)
        reinterpret_cast<P*>(bf16_output)[tid[k]] = o[k];
}
a = lamport_wave_max_bits(a);                             // wave 内：整数 max，DPP
if(threadIdx.x % kLamportWave == 0)
    __hip_atomic_fetch_max(&amax_bits, a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
lamport_lds_barrier();                                    // 第 2 次（也是最后一次）同步
const float row_amax = __builtin_bit_cast(float, amax_bits);
const float fp8_max  = opus::cast<opus::fp32_t>(opus::numeric_limits<OutT>::max());
const float scale    = row_amax * (float)(1. / fp8_max);
const float inv      = __builtin_amdgcn_rcpf(scale);
```

wave 内的整数 max（`新:1556`）：

```cpp
DINLINE uint32_t lamport_wave_max_bits(uint32_t p)
{
#define LAMPORT_MAX_STEP(OFF)                                                     \
    {                                                                             \
        const uint32_t q = (uint32_t)lamport_xor_lane_i<OFF>((int)p);             \
        p                = p > q ? p : q;                                         \
    }
    LAMPORT_MAX_STEP(16) LAMPORT_MAX_STEP(8) LAMPORT_MAX_STEP(4) LAMPORT_MAX_STEP(2)
    LAMPORT_MAX_STEP(1)
#undef LAMPORT_MAX_STEP
    return p;
}
```

量化部分（scale、rcp、截断、转 FP8）和旧版完全相同，保证逐位一致。

### 8.6 执行过程

```
0 号线程：amax_bits = 0
平方和规约（同步①）→ 所有 wave 都看到 amax_bits = 0，也拿到 inv_rms
每个线程：算出 PACKS 个输出包 o[k]
         每个包内 8 个值取 |x| 的浮点 max（和 car_fused 相同），转成位
         几个包之间整数 max → a
每个 wave：5 步整数 max（permlanex16 + 4 次 DPP）→ 32 个 lane 都得到本 wave 的最大值
         lane 0：LDS 原子 max(amax_bits, a)
同步② → 所有线程读 amax_bits（同一地址，LDS 广播）→ 转回 float → scale、inv → 量化、写输出
```

### 8.7 指令拆解

新版（`new_fp8_split`，节选）：

```
ds_store_b32 v1, v1 offset:128            ; 0 号线程：amax_bits = 0（在同步①之前）
...
v_max3_num_f32 v10, |v6|, 0, |v7|         ; 一个包内：max(|x0|, 0, |x1|)
v_max3_num_f32 v1, v10, |v2|, |v3|        ; 再并入 |x2|, |x3| ...
v_max3_num_f32 v1, v1, |v8|, |v9|
v_max3_num_f32 v1, v1, |v4|, |v5|
v_mov_b32_e32 v10, v1
s_mov_b32 s0, 0x76543210
v_permlanex16_b32 v10, v10, s0, 0xfedcba98    ; 取另一半（lane xor 16）的值
v_max_u32_e32 v1, v1, v10                       ; 整数 max
v_max_u32_dpp v1, v1, v1 row_xmask:8 row_mask:0xf bank_mask:0xf bound_ctrl:1   ; 取 lane xor 8 并 max，一条指令
v_max_u32_dpp v1, v1, v1 row_xmask:4 ...
v_max_u32_dpp v1, v1, v1 row_xmask:2 ...
v_mov_b32_dpp v10, v1 row_xmask:1 ...
v_max_u32_e32 v1, v1, v10
s_ctz_i32_b32 s15, s7                     ; ↓ 编译器的原子优化：在活跃 lane 中循环
v_readlane_b32 s16, v1, s15
s_max_u32 s6, s6, s16
...
ds_max_u32 v1, v10 offset:128             ; LDS 原子 max，写进 amax_bits
...
s_barrier_signal -1                       ; 同步②
s_barrier_wait 0xffff
ds_load_b32 v14, v14 offset:128           ; 所有线程读 amax_bits
...
v_rcp_f32_e32 v10, v1                     ; inv = 1 / scale（近似倒数）
v_med3_num_f32 v6, v6, -v11, v11          ; 截到 [-fp8_max, fp8_max]
v_cvt_pk_fp8_f32 v12.l, v6, v7            ; 两个 fp32 转两个 FP8，拼进 16 位
```

| 指令 / builtin | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `v_max3_num_f32 d, a, b, c` | 三个数取最大，`\|a\|` 表示取绝对值 | 一个包内 8 个值的 \|x\| 最大值 | 和 car_fused 相同的逐元素步骤，保证结果一致 |
| `__builtin_bit_cast(uint32_t, f)` | 按位重新解释，不产生指令 | float → 无符号整数 | 后面全用整数比较 |
| `v_permlanex16_b32` | 每个 lane 从另一组 16 个 lane 里按选择码取值（第 9 节详细讲） | 取 lane xor 16 的值 | wave 内最大值的第 1 步 |
| `v_max_u32_dpp ... row_xmask:N` | 无符号整数 max，其中一个操作数通过 DPP 取自 lane xor N | wave 内最大值的第 2~4 步 | 取值和 max 合成一条指令 |
| `__hip_atomic_fetch_max(p, v, relaxed, workgroup)` → `ds_max_u32` | LDS 上的原子 max：`*p = max(*p, v)`，硬件保证多个 wave 同时做也不会丢 | 每个 wave 的结果合进 amax_bits | 代替"写部分值数组 + 第二轮规约 + 第二次同步" |
| `s_ctz_i32_b32` / `v_readlane_b32` / `s_max_u32` 循环 | 编译器的原子优化：先在 wave 内把所有活跃 lane 的值合成一个，再由一个 lane 做原子操作 | 这里只有 lane 0 活跃，循环只跑一次 | 编译器自动生成，开销很小 |
| `ds_load_b32 ... offset:128` | 读 LDS | 所有线程读同一个地址 | LDS 对同一地址的读是广播，没有冲突 |
| `__builtin_amdgcn_rcpf` → `v_rcp_f32` | 近似倒数 | inv = 1 / scale | 和 car_fused 相同 |
| `__builtin_amdgcn_fmed3f(x, -m, m)` → `v_med3_num_f32` | 三个数取中间值 | 等价于把 x 截到 [-m, m] | 一条指令完成截断 |
| `__builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi)` → `v_cvt_pk_fp8_f32` | 两个 fp32 转成两个 FP8，放进 16 位 | 量化 | 和 car_fused 相同 |

旧版对照（`old_fp8_row`，节选）：

```
ds_bpermute_b32 v11, v14, v10     ; 取 lane xor 16
s_wait_dscnt 0x0
v_max_num_f32_e32 v11, v11, v11   ; 规范化（处理 NaN），每步都要
v_max_num_f32_e32 v10, v10, v11   ; 浮点 max
...                               ; 共 5 轮
ds_store_b32 v19, v10 offset:128
s_wait_storecnt_dscnt 0x0         ; 同步③，还等跨卡写确认
s_barrier_signal -1
...                               ; 第 0 个 wave 再做 5 轮 ds_bpermute
ds_store_b32 v10, v0 offset:260
s_barrier_signal -1               ; 同步④
```

### 8.8 通用性和经验

- **非负、无 NaN 的 float 求最大值，可以直接按整数比较。** 常见场景：量化的 amax、|x| 的最大值、范数相关的最大值。前提要自己证明清楚（这里是"从 0 开始、取绝对值、fmaxf 丢 NaN"）。有负数时不能直接用（负数的位顺序是反的）。
- **结果和顺序无关的规约（max、min、按位或/与），跨 wave 可以用 LDS 原子操作**，省掉部分值数组和第二轮规约。**浮点求和不行**，原子加的顺序不固定，结果不能逐位复现。
- **把"准备工作"挂在已有的同步上**：清零放在前一次同步之前，就不用为它单独同步一次。设计规约时先数一数总共要几次同步，看能不能合并。
- 量化部分（scale、rcp、clamp、cvt）一点没动，保证和原实现逐位一致。只改"怎么汇总"，不改"算什么"。

---

## 9. Tip 7：wave 内求和用 DPP，跨 wave 只同步一次

### 9.1 提升多少

- POC 阶段，2048×1：bf16 5.65 → 5.51，FP8 6.17 → 5.73。这是"DPP + 少一次同步"合在一起的效果。FP8 收益更大，因为它有两次规约。
- 后来又打开了 `bound_ctrl`，让取值和加法合成一条指令。这一步和 FP8 amax（第 8 节）在同一次修改里，没有单独测。
- 收益比第 4 节小：第 4 节省的是一次跨卡来回（约 1 us），这里省的是 LDS 的几次往返（几十到上百纳秒）。

### 9.2 原来哪里慢、怎么发现的

**wave 内求和的"蝴蝶"做法**：32 个 lane 各有一个数，分 5 轮，每轮每个 lane 拿"编号异或 N"的那个 lane 的数加到自己身上：

```
第 1 轮 N=16：lane i 加上 lane i^16 的数 → 每个 lane 手里是 2 个数的和
第 2 轮 N=8 ：                            → 4 个数的和
第 3 轮 N=4 ：                            → 8 个
第 4 轮 N=2 ：                            → 16 个
第 5 轮 N=1 ：                            → 32 个，所有 lane 手里都是同一个总和
```

旧版"拿别的 lane 的数"用 `__shfl_xor`，在 gfx1250 上编译成 `ds_bpermute_b32`。看汇编，每一轮是：

```
v_xor_b32_e32 v15, 8, v16          ; 算目标 lane 号 = lane ^ 8
v_cmp_lt_u32_e32 vcc_lo, v15, v17  ; 检查目标 lane 是否越界（__shfl_xor 的语义要求）
v_cndmask_b32_e32 v15, v16, v15    ; 越界就取自己
v_lshlrev_b32_e32 v14, 2, v15      ; lane 号 × 4 = 字节地址
ds_bpermute_b32 v14, v14, v1       ; 通过 LDS 的交换电路取值
s_wait_dscnt 0x0                   ; 等它回来
v_add_f32_e32 v1, v1, v14          ; 加
```

一轮 7 条指令，其中 `ds_bpermute` 要走 LDS 流水线，几十个时钟；5 轮严格串行。

**跨 wave**：旧版同步一次后，只有 0 号线程把 8~32 个部分和串行加完，写进 LDS 的 `result`，再同步一次让大家读。两次同步，中间其他 wave 都在空等。

**怎么发现的**：看汇编，规约这一段全是 `ds_bpermute` + `s_wait_dscnt` 的串行链。gfx1250 是 wave32，有 DPP 可以在寄存器之间直接取值，不用经过 LDS。

### 9.3 怎么改、为什么这样改

**wave 内：换成寄存器之间直接取值的指令。** wave32 的 32 个 lane 被硬件分成两"行"（row），每行 16 个 lane：

```
row 0：lane 0 ~ 15
row 1：lane 16 ~ 31
```

- **xor 1、2、4、8**：源 lane 和自己在同一行里。用 DPP 的 `row_xmask:N`，每个 lane 读本行内 `lane ^ N` 的值。
- **xor 16**：源 lane 在另一行，DPP 做不到，用 `v_permlanex16_b32`（跨行取值）。

配对关系和 `__shfl_xor` 完全一样，加的顺序也一样，所以结果逐位一致。

**`bound_ctrl = true`**：DPP 有一个选项，决定"源 lane 无效时怎么办"：false 表示保留目标寄存器的旧值，true 表示读 0。`row_xmask` 的源 lane 永远有效，所以这个选项不改变结果。但设成 true 之后，编译器就不用保留"旧值"，可以把"取值"和"加法"合成一条 `v_add_f32_dpp`。

**跨 wave：每个线程自己加。** 同步一次后，每个线程都按 0、1、2…… 的顺序把所有部分和读出来自己加。

- 所有线程读的是同一个地址，LDS 会把一次读广播给所有 lane，没有冲突。
- 每个线程得到同一个结果，不需要第二次同步来分发。
- 加的顺序和 car_fused 的 0 号线程完全一样，逐位一致。

**试过但更慢的写法**：用 `readlane` 在标量单元上串行加。一个 CU 上所有 wave 共用一个 SALU，几十个 wave 同时这么做会排队。

### 9.4 原来的代码

`block_allreduce_f32`（`旧:816`），完整代码见第 4.4 节：

```cpp
for(int off = warpSize / 2; off > 0; off >>= 1)
{
    const float o = __shfl_xor(v, off);        // → ds_bpermute_b32
    v             = kMax ? fmaxf(v, o) : v + o;
}
if(threadIdx.x % warpSize == 0)
    partial[threadIdx.x / warpSize] = v;
__syncthreads();
...
else if(threadIdx.x == 0)                      // 只有 0 号线程加
{
    float r = partial[0];
    for(int w = 1; w < (int)(blockDim.x / warpSize); ++w)
        r = kMax ? fmaxf(r, partial[w]) : r + partial[w];
    result = r;
}
__syncthreads();                               // 第二次同步，分发结果
return result;
```

### 9.5 现在的代码

取值（`新:1523`）：

```cpp
template <int OFF>
DINLINE int lamport_xor_lane_i(int i)
{
#if AITER_GFX1250_HAVE_B128
    if constexpr(OFF == 16)
        return __builtin_amdgcn_permlanex16(i, i, 0x76543210, 0xfedcba98, false, false);
    else
        return __builtin_amdgcn_update_dpp(i, i, 0x160 + OFF, 0xf, 0xf, true); // row_xmask:OFF
#else
    return __shfl_xor(i, OFF); // other archs only build the kernel, never run it
#endif
}

template <int OFF>
DINLINE float lamport_xor_lane(float v)
{
    return __builtin_bit_cast(float, lamport_xor_lane_i<OFF>(__builtin_bit_cast(int, v)));
}
```

wave 内求和（`新:1544`）：

```cpp
DINLINE float lamport_wave_sum(float p)
{
    p = p + lamport_xor_lane<16>(p);
    p = p + lamport_xor_lane<8>(p);
    p = p + lamport_xor_lane<4>(p);
    p = p + lamport_xor_lane<2>(p);
    p = p + lamport_xor_lane<1>(p);
    return p;
}
```

跨 wave（`lamport_row_sum` 的后半段，`新:1586`）：

```cpp
lamport_lds_barrier();
// every lane reads the same address (an LDS broadcast) and adds on the
// VALU: a readlane loop would queue every wave of the CU on its SALU
float r = partial[0];
for(int g = 1; g < row_groups; ++g)
    r += partial[g];
return r;
```

### 9.6 执行过程（一个 wave）

```
5 步，全部在寄存器里：
  v_permlanex16 取对面行同位置的值 → 加     （xor 16）
  v_add_f32_dpp row_xmask:8               （取值 + 加，一条指令）
  v_add_f32_dpp row_xmask:4
  v_add_f32_dpp row_xmask:2
  v_mov_b32_dpp row_xmask:1 + 加           （最后一步编译器选择拆开，和别的指令双发射）
lane 0 写 partial[w] → 一次同步 → 每个线程读 partial[0..n-1]，按顺序加
```

### 9.7 指令拆解

新版（`new_bf16_row1`）：

```
v_mov_b32_e32 v15, v14
s_mov_b32 s0, 0x76543210
v_permlanex16_b32 v15, v15, s0, 0xfedcba98          ; v15 = 对面行同位置 lane 的 v14
v_add_f32_e32 v14, v14, v15
v_add_f32_dpp v14, v14, v14 row_xmask:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32_dpp v14, v14, v14 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_add_f32_dpp v14, v14, v14 row_xmask:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
v_mov_b32_dpp v15, v14 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
...
v_dual_add_f32 v14, v14, v15 :: v_dual_lshrrev_b32 v0, 3, v0   ; 两条独立运算打包成一条
ds_store_b32 v0, v14                                             ; partial[w]
s_wait_dscnt 0x0
s_barrier_signal -1                                              ; 唯一一次同步
s_barrier_wait 0xffff
ds_load_b32 v0, v0                                               ; partial[0]
ds_load_2addr_b32 v[14:15], v20 offset0:1 offset1:2              ; 一次读两个：partial[1]、[2]
...
v_add_f32_e32 v0, v0, v14                                        ; 按顺序加
```

| 指令 / builtin | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| **DPP**（数据并行原语） | 不是独立指令，而是附在 VALU 指令上的修饰：其中一个源操作数不读本 lane 的寄存器，而是按规则读别的 lane 的 | 让加法的一个操作数直接来自别的 lane | 不经过 LDS，没有额外延迟 |
| `row_xmask:N` | DPP 规则之一：lane i 读同一行内 lane (i xor N) 的值。编码是 0x160 + N | xor 1/2/4/8 | 蝴蝶的后 4 步 |
| `row_mask:0xf` / `bank_mask:0xf` | 哪些行 / 哪些 bank（每 4 个 lane 一组）参与写结果，0xf 表示全部 | 全部 lane 都写 | — |
| `bound_ctrl:1` | 源 lane 无效时读 0（为 0 时保留目标的旧值） | 这里源 lane 永远有效，不影响结果 | 让编译器能合成 `v_add_f32_dpp` |
| `__builtin_amdgcn_update_dpp(old, src, ctrl, row_mask, bank_mask, bound_ctrl)` | 生成一条带 DPP 的 `v_mov_b32_dpp`；old 是源无效时保留的值 | 取 lane ^ N 的值 | 编译器再把这条 mov 和后面的加合并 |
| `v_permlanex16_b32 vD, vS, sel_lo, sel_hi` | 每个 lane 从**另一行**取值，取哪个位置由选择码决定：sel_lo 的第 i 个 4 位给 lane i（i<8），sel_hi 给 lane 8~15 | 选择码 0x76543210 / 0xfedcba98 表示"位置不变，换到另一行"，等于 xor 16 | 蝴蝶的第 1 步 |
| `__builtin_amdgcn_permlanex16(old, src, sel_lo, sel_hi, fi, bound_ctrl)` | 生成上面这条指令 | 同上 | — |
| `ds_bpermute_b32 vD, vAddr, vData`（旧） | 每个 lane 按 vAddr/4 指定的 lane 号取 vData。借用 LDS 的交换电路，不占 LDS 空间 | 旧版 `__shfl_xor` | 要走 LDS 流水线，还要 `s_wait_dscnt` |
| `v_dual_add_f32 ... :: v_dual_lshrrev_b32 ...` | VOPD 双发射：两条互不依赖的运算打包成一条指令 | 编译器自动做的 | 少占一个指令周期 |
| `ds_load_2addr_b32 v[a:b], vAddr offset0:x offset1:y` | 一条指令从两个偏移各读一个 dword | 一次读两个部分和 | 少发一半 LDS 读 |

### 9.8 通用性和经验

- **适用于 wave32 的架构**（gfx10 以后的 RDNA、gfx12、gfx1250）。gfx942（MI300）是 wave64，DPP 的规则集不同（有 row_shr、row_bcast 等，没有这里的写法），不能照搬。
- **前提**：参与的 lane 全部活跃。DPP 从没有执行的 lane 取值会拿到旧值。
- **检查方法**：看汇编里有没有单独的 `v_mov_b32_dpp` 后面紧跟一条加法。如果有，试试 `bound_ctrl = true`（前提是源 lane 一定有效）。
- **跨 wave 的串行和**，用"每个线程从 LDS 广播读、自己在 VALU 上加"，代替"一个线程算完再同步一次分发"。不要用 `readlane` 在 SALU 上做，CU 上所有 wave 会挤在一个 SALU 上。
- **只要规约在关键路径上才值得做。** kernel 如果卡在访存吞吐上，看不出收益。

---

## 10. Tip 8：跨卡读写改用 buffer 指令，加 BYPASS 缓存策略

### 10.1 提升多少

约 1~3%（移植阶段），在测量噪声附近。保留它主要是因为：和 Gluon 的访存语义一致，汇编更简洁（偏移变成 32 位）。

### 10.2 原来哪里慢、怎么发现的

**怎么发现的**：读 Gluon kernel 编译出来的汇编（第 14.2 节的方法），它的跨卡读写是：

```
buffer_load_b128 v[46:49], v99, s[36:39], null offen th:TH_LOAD_BYPASS scope:SCOPE_SYS
buffer_store_b128 ...                                th:TH_STORE_BYPASS scope:SCOPE_SYS
```

而旧版是：

```
global_store_b128 v32, v[10:13], s[22:23] scale_offset scope:SCOPE_SYS
global_load_b128 v[26:29], v[34:35], off scope:SCOPE_SYS
```

区别是 Gluon 多了 `th:TH_*_BYPASS`（缓存策略：绕过缓存）。

**为什么旧版设不了**：旧版用的 `__builtin_amdgcn_global_store_b128(ptr, v, "")` 只有一个字符串参数，用来指定同步范围（scope），没有地方填缓存策略。buffer 指令的 builtin 最后一个参数可以填完整的缓存策略。

**为什么 BYPASS 合理**：

- 写给别的卡的数据，本卡以后不会再读，在本卡缓存里留一份没用。
- 轮询读的格子会被别的卡改写，每次都必须读到最新值，缓存它也没用。

### 10.3 怎么改、为什么这样改

```cpp
lamport_store_sys(slot_base, pack, v)   // 写：buffer_store_b128，策略 27
lamport_load_sys(slot_base, pack)       // 读：buffer_load_b128，策略 27
```

**策略值 27 的含义**：

```
27 = 0b11011
     低 3 位  TH    = 3（BYPASS）
     第 3~4 位 SCOPE = 3（SYS，整个系统可见，包括别的卡）
```

**buffer 指令的地址怎么给**：buffer 指令的地址 = 描述符里的基址 + 32 位偏移。

- 基址：`lamport_slot()` 算出"某一块缓冲、某个发送方"那一格的起始地址。它对整个 wave 相同，放在 SGPR。
- 偏移：包号 × 16，放在 VGPR。

偏移从"一格"的起点算，而不是从整个收件箱的起点算。一格最多 slot 个包，远小于 2 GiB，所以 32 位一定够；整个收件箱（3 × ngpus 格）在容量设得很大时可能超过。

### 10.4 原来的代码

`旧:1487`：

```cpp
DINLINE void lamport_store_b128(void* dst, lamport_u32x4 v)
{
#if AITER_GFX1250_HAVE_B128
    __builtin_amdgcn_global_store_b128((llx_v4u_gptr)dst, v, "");
#else
    __builtin_nontemporal_store(v, reinterpret_cast<lamport_u32x4*>(dst));
#endif
}
// 调用：每个线程自己算 64 位地址
P* dst = reinterpret_cast<P*>(peers.ptrs[(rank + i) % ngpus]) + (round * ngpus + rank) * slot + tid;
lamport_store_b128(dst, local.i);
```

### 10.5 现在的代码

`新:1464`：

```cpp
// Slot (buf, sender) of an inbox, inbox[buf][sender][slot] of 16B packs. The
// caller passes uniform values, so the result is in SGPRs.
DINLINE char* lamport_slot(void* inbox, uint32_t buf, int sender, int ngpus, int slot)
{
    return static_cast<char*>(inbox) + (int64_t)(buf * ngpus + sender) * slot * 16;
}

DINLINE void lamport_store_sys(char* slot_base, int pack, lamport_u32x4 v)
{
#if AITER_GFX1250_HAVE_B128
    __builtin_amdgcn_raw_buffer_store_b128(
        v, __builtin_amdgcn_make_buffer_rsrc(slot_base, 0, 0x7fffffff, 0), pack * 16, 0, 27);
#else
    lamport_store_b128(reinterpret_cast<lamport_u32x4*>(slot_base) + pack, v);
#endif
}

DINLINE lamport_u32x4 lamport_load_sys(char* slot_base, int pack)
{
#if AITER_GFX1250_HAVE_B128
    return __builtin_amdgcn_raw_buffer_load_b128(
        __builtin_amdgcn_make_buffer_rsrc(slot_base, 0, 0x7fffffff, 0), pack * 16, 0, 27);
#else
    return lamport_load_b128(reinterpret_cast<lamport_u32x4*>(slot_base) + pack);
#endif
}
// 调用
lamport_store_sys(lamport_slot(all[g], round[k], rank, ngpus, slot), tid[k], local[k].i);
```

### 10.6 执行过程

```
标量单元：slot_base = 收件箱基址 + (缓冲号 × ngpus + 发送方) × slot × 16   （整个 wave 算一次）
         用 slot_base 拼出 4 个 SGPR 的描述符
向量单元：offset = 包号 × 16                                              （每个 lane 一个）
发出 buffer_store_b128：地址 = 描述符基址 + offset，策略 BYPASS + SYS
```

### 10.7 指令拆解

新版（`new_bf16_row1`）：

```
s_add_nc_u64 s[34:35], s[18:19], s[2:3]         ; 基址 = p0 + 格子偏移（标量运算）
s_or_b64 s[36:37], s[34:35], 0xfe00000000000000 ; 描述符前 64 位：基址，最高 7 位放记录数的低 7 位
s_mov_b32 s38, 0xffffff                         ; 描述符第 3 个 dword：记录数的高 24 位
                                                ; （s39 = 0：格式和标志位）
v_lshlrev_b32_e32 v1, 4, v26                    ; offset = 包号 × 16
buffer_store_b128 v[10:13], v1, s[36:39], null offen th:TH_STORE_BYPASS scope:SCOPE_SYS
```

| 指令 / builtin | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `__builtin_amdgcn_make_buffer_rsrc(base, stride, num_records, flags)` | 生成 buffer 描述符（V#）：4 个 SGPR，包含基址、步长、记录数（访问上限）、格式标志 | base = 一格的起点；stride 0；num_records = 0x7fffffff（不限制）；flags 0 | buffer 指令必须通过描述符访问内存 |
| 描述符的拼法（从汇编看） | num_records 0x7fffffff 被拆成两段：低 7 位放进前 64 位的最高 7 位（0xfe00000000000000），高 24 位放进第 3 个 dword（0xffffff） | 3 条标量指令拼出描述符 | 整个 wave 只做一次 |
| `__builtin_amdgcn_raw_buffer_store_b128(data, rsrc, voffset, soffset, aux)` | 按描述符写 16 字节。voffset 来自 VGPR，soffset 来自 SGPR（这里是 0），aux 是缓存策略 | 跨卡写和回填 | aux = 27 设 BYPASS + SYS |
| `buffer_store_b128 vdata, voffset, s[rsrc], null offen ...` | `offen` 表示使用 VGPR 偏移；`null` 表示没有 SGPR 偏移 | 同上 | — |
| `th:TH_STORE_BYPASS` / `th:TH_LOAD_BYPASS` | 缓存策略（temporal hint）：BYPASS 表示不在缓存里保留这次访问的数据 | 写给别的卡的数据、轮询读的数据 | 都不会被本卡复用 |
| `scope:SCOPE_SYS` | 这次访问对整个系统（包括其他 GPU）可见 | 跨卡协议必须 | 新旧版都有 |
| `global_store_b128 ... scope:SCOPE_SYS`（旧） | 按 64 位地址写 16 字节 | 旧版跨卡写 | 设不了 `th` |

### 10.8 通用性和经验

- **想设缓存策略，但 builtin 不支持时，换成 buffer 指令的 builtin**，最后一个参数可以填完整的策略。
- **BYPASS 适合"本 CU 不会再读"的数据**：写给别的卡的数据、只读一次的数据。是否真的变快要实测，这里收益很小。
- **偏移用 32 位、基址用标量**：地址计算从每个 lane 的 64 位运算，变成整个 wave 一次标量运算加每个 lane 一个 32 位乘法。前提是保证偏移不会超过 32 位（这里按"一格"算）。
- **读竞争对手的汇编是找优化点的好办法**：同样的硬件，别人用了什么指令和选项，一比就知道。

---

## 11. Tip 9：轮询只读另外 3 张卡

### 11.1 提升多少

没有单独测准。（POC 阶段测的时候正好有别的容器在用 GPU，数据不可信。）每轮轮询的读请求从 4 个变成 3 个（TP4），TP2 从 2 个变成 1 个。

### 11.2 原来哪里慢

旧版每轮轮询读 ngpus 个格子，包括自己那一格。自己那格永远没人写，永远是 0xffff，读回来再用 `g == rank ? 0 : gm` 屏蔽掉：

```cpp
for(int g = 0; g < ngpus; ++g)
    scratch[g].i = lamport_load_b128(inbox + g * slot);   // 包括 g == rank
...
m |= g == rank ? 0u : gm;                                  // 自己那格不算
```

这样写的原因（注释里写着）：如果用 `if(g != rank)` 跳过，编译器会生成分支，每次读都要等上一次的分支判断，读就不能一次全部发出。

### 11.3 怎么改、为什么这样改

只读 rank+1、rank+2、rank+3（模 ngpus）这 3 格，用编译期展开的下标 i，实际卡号 g = (rank + 1 + i) % ngpus。g 对整个 wave 相同，地址偏移都在 SGPR 里，不需要分支，也不增加每个 lane 的地址计算。

求和时必须按卡号 0、1、2、3 的顺序加（逐位一致的要求），所以要把数据放回原来的顺序：卡 g 的数据在 `scratch[(g - rank - 1 + ngpus) % ngpus]`；自己的数据用本卡的 input。

### 11.4 原来的代码

`旧:1568`：

```cpp
const P* inbox = reinterpret_cast<const P*>(peers.ptrs[rank]) + round * ngpus * slot + tid;
pack scratch[ngpus];
bool pending;
do
{
    asm volatile("" ::: "memory");
    for(int g = 0; g < ngpus; ++g)
        scratch[g].i = lamport_load_b128(inbox + g * slot);
    __builtin_amdgcn_sched_barrier(0);
    uint32_t m = 0;
    for(int g = 0; g < ngpus; ++g)
    {
        uint32_t gm = 0;
        for(int j = 0; j < 4; ++j)
            gm |= lamportSentinelMask(scratch[g].i[j]);
        m |= g == rank ? 0u : gm;
    }
    pending = m != 0;
} while(pending);
// 求和
for(int g = 0; g < ngpus; ++g)
{
    const P v = g == rank ? local.data : scratch[g].data;
    for(int j = 0; j < pack_size; ++j)
        x[j] += upcast_s(v[j]);
}
```

### 11.5 现在的代码

`新:1746`：

```cpp
// the peers only, rank + 1 .. rank + ngpus - 1: the slot offsets are
// uniform (SGPR), so this costs no extra address math per load
pack scratch[PACKS][ngpus - 1];
bool pending;
do
{
    // without the clobber the loads are hoisted out of the loop
    asm volatile("" ::: "memory");
    for(int k = 0; k < PACKS; ++k)
        for(int i = 0; i < ngpus - 1; ++i)
        {
            const int g     = (rank + 1 + i) % ngpus;
            scratch[k][i].i =
                lamport_load_sys(lamport_slot(own, round[k], g, ngpus, slot), tid[k]);
        }
    // issue all loads before any mask math
    __builtin_amdgcn_sched_barrier(0);
    uint32_t m = 0;
    for(int k = 0; k < PACKS; ++k)
        for(int i = 0; i < ngpus - 1; ++i)
            for(int j = 0; j < 4; ++j)
                m |= lamportSentinelMask(scratch[k][i].i[j]);
    pending = m != 0;
} while(pending);
// 求和：按卡号顺序
for(int g = 0; g < ngpus; ++g)
{
    const int at = (g - rank - 1 + ngpus) % ngpus; // ngpus - 1: this rank
    P v          = local[k].data;
    for(int i = 0; i < ngpus - 1; ++i)
        v = at == i ? scratch[k][i].data : v;      // 常量下标逐个比较，不用运行时下标
    for(int j = 0; j < pack_size; ++j)
        x[k][j] += upcast_s(v[j]);
}
```

### 11.6 执行过程

```
每轮：
  一次发出 3 条 buffer_load（3 张别的卡的格子）
  等第 1 条回来就开始检查 0xffff，第 2、3 条回来再接着查
  只要有一个半字是 0xffff → 再来一轮
全部到齐后：按卡号 0..3 依次取 本卡 input 或 scratch[at]，fp32 累加
```

### 11.7 指令拆解

新版（`new_bf16_row1`）：

```
s_clause 0x2                                   ; 下面 3 条访存指令作为一组连续发出
buffer_load_b128 v[22:25], v1, s[0:3], null offen th:TH_LOAD_BYPASS scope:SCOPE_SYS
buffer_load_b128 v[18:21], v1, s[24:27], null offen th:TH_LOAD_BYPASS scope:SCOPE_SYS
buffer_load_b128 v[14:17], v1, s[20:23], null offen th:TH_LOAD_BYPASS scope:SCOPE_SYS
s_wait_loadcnt 0x2                             ; 等第 1 条回来（还剩 2 条没回来也可以继续）
v_or_b32_e32 v28, 0x80008000, v23              ; ↓ 检查有没有 0xffff 半字
v_add_nc_u32_e32 v28, 0x80008001, v28
v_and_b32_e32 v28, v28, v23
...
s_wait_loadcnt 0x1                             ; 第 2 条
...
s_wait_loadcnt 0x0                             ; 第 3 条
...
v_cmp_eq_u32_e32 vcc_lo, 0, v28                ; 没有 0xffff 的 lane 记下来
s_and_not1_b32 exec_lo, exec_lo, s19           ; 这些 lane 退出循环，其余继续
```

| 指令 / builtin | 意思 | 这里做什么 | 目的 |
|---|---|---|---|
| `asm volatile("" ::: "memory")` | 空汇编，声明"可能改了任何内存" | 放在循环开头 | 阻止编译器把读提到循环外（否则它会认为每轮读的都一样） |
| `__builtin_amdgcn_sched_barrier(0)` | 调度屏障：编译器不能把任何指令跨过这里重排 | 放在"发出所有读"和"检查"之间 | 先把读全部发出，再开始检查，读的延迟可以重叠 |
| `s_clause 0x2` | 提示硬件接下来 3 条访存连续发出，不要插入别的 wave 的访存 | 3 条轮询读 | 编译器自动生成 |
| `s_wait_loadcnt N` | 等到未返回的读不超过 N 个 | 按顺序逐个等 | 先回来的先检查 |
| `s_and_not1_b32 exec_lo, exec_lo, s19` | 修改 exec 掩码（哪些 lane 继续执行） | 已经收齐的 lane 退出循环 | 每个 lane 独立判断 |

### 11.8 通用性和经验

- **先数清楚每次循环发出了哪些访存，有没有"读了再扔掉"的。** 这是最容易拿到的优化，没有任何风险。
- **用"统一下标"避免分支**：这里的技巧是把"跳过自己"变成"从自己的下一张卡开始数 ngpus−1 张"，下标 i 是编译期常量，卡号 g 对整个 wave 相同，不需要分支。
- **数组要留在寄存器里，就只用常量下标访问**：按卡号重排时，用展开的比较逐个选，而不是 `scratch[at]` 这样的运行时下标（那样数组会被放进 scratch 内存）。

---

## 12. Tip 10：host 端按形状自动选模式

### 12.1 提升多少

在 TP2/TP4、hidden 1024~8192、M 1~2048、bf16 和 FP8 上，和"每个形状取 6 种固定配置里最快的"相比，自动选择平均只慢 0.1%（TP4）和 1.4%（TP2）。

### 12.2 原来哪里慢、怎么得出规则的

split 和各种 PACKS 各有擅长的形状（第 3、5 节），选错了会慢很多。例如 6144×1 用 row PACKS=1 是 8.45，用 split 是 5.95。

**怎么得出规则**：

1. 用临时的环境变量开关，在同一个进程里切换 6 种固定配置：split（发送块 256 / 128 个包）、row（PACKS = 1 / 2 / 4 / 8）。
2. 扫 TP2/TP4 × 6 个 hidden × M 1~2048 × bf16/FP8，每个形状记下最快的配置。
3. 观察规律，写成尽量简单的规则。
4. 用同一批数据检验：规则选出来的配置，和每个形状最快的配置差多少。

### 12.3 规则

```
split：  一行超过 128 个包（hidden > 1024）
         且 TP4 时 M ≤ 64；TP2 时总包数 ≤ 32768（M × hidden / 8）
row：    其余情况。PACKS 取 {1, 2, 4, 8} 中满足下面两条的最小值：
           - 整除一行的组数（一行的包数 / 32）
           - block（= 一行包数 / PACKS）不超过上限
         上限：M ≤ 256 时 256 线程；M ≤ 512 时 128 线程；更大时 64 线程
```

后两档上限是因为 M 大时 block 数远超 CU 数，小 block 能让一个 CU 同时跑几个，互相掩盖等待。

### 12.4 原来的代码

见第 3.4 节 `cluster_threads()`：行数少于 CU 数、一行超过 256 包时走 cluster，否则一行一个 block、一个线程一个包。

### 12.5 现在的代码

`新:2500`：

```cpp
static int row_block_packs(int row_packs, int64_t rows)
{
    const int most   = rows <= 256 ? 256 : rows <= 512 ? 128 : 64;
    const int groups = row_packs / kLamportWave;
    int packs        = 1;
    for(int p : {1, 2, 4, 8})
    {
        if(groups % p != 0)
            continue;
        packs = p;
        if(row_packs / p <= most)
            break;
    }
    return packs;
}
```

`新:2551`：

```cpp
const bool split =
    row_packs > 128 && (world_size >= 4 ? rows <= 64 : packs <= 32768);
const int packs_per_thread = split ? 1 : row_block_packs(row_packs, rows);
```

### 12.6 执行过程（几个例子，TP4，bf16）

| 形状 | 一行包数 | 模式 | PACKS | block 线程数 | grid |
|---|---|---|---|---|---|
| 2048×1 | 256 | split | 1 | 256 | 1 个发送块 + 1 行 |
| 6144×1 | 768 | split | 1 | 768 | 3 + 1 |
| 6144×64 | 768 | split | 1 | 768 | 192 + 64 |
| 6144×128 | 768 | row | 4 | 192 | 128 |
| 8192×256 | 1024 | row | 4 | 256 | 256 |
| 8192×512 | 1024 | row | 8 | 128 | 512 |
| 1024×1 | 128 | row | 1 | 128 | 1 |

### 12.7 指令拆解

host 端代码，没有特殊指令。kernel 按 PACKS 和 SPLIT 实例化了 5 种（split PACKS=1；row PACKS=1/2/4/8），每种都会被选到。

### 12.8 通用性和经验

- **先扫全量、再写规则、最后用"最优配置"做基准检验规则**。规则好不好，不看它在几个形状上的表现，看它和"每个形状的最优"差多少。
- **规则只依赖所有 rank 都相同的量**（行数、hidden、world_size）。多卡 kernel 里，每张卡必须选同一种 grid，否则收件箱的布局对不上。
- 规则尽量简单，例外越少越好。扫描的数据要保留，换硬件时重扫。

---

## 13. Tip 11：禁止乘加合并 `#pragma clang fp contract(off)`

### 13.1 提升多少

不提速。它保证改了代码结构之后，结果仍然和 car_fused 逐位一致。

### 13.2 原来的问题、怎么发现的

加了 PACKS（第 5 节）以后，逐位比对测试在 6144×300 的第 68 行发现 3 个元素和 car_fused 差一个最低位，residual_out 完全一致。

对比汇编：

- car_fused 算平方和：先 `v_pk_mul_f32` 算平方（舍入一次），再 `v_add_f32` 加（再舍入一次）。
- 新代码换了形状后，编译器把其中一部分"乘再加"合成了 `v_pk_fma_f32`：乘完不舍入，直接加，只舍入一次。结果就差了一点。

合不合成，是编译器看代码形状决定的。HIP 默认允许跨语句合成（`-ffp-contract=fast`），所以 car_fused 是"碰巧"没合成。

### 13.3 怎么改

在 kernel 函数体最开头加 `#pragma clang fp contract(off)`，禁止这个 kernel 里的乘加合成。放在函数体外或别的位置会编译报错。加上以后 PACKS = 1/2/4/8 全部逐位一致。

### 13.4 原来的代码

旧版 kernel 里没有这条 pragma，PACKS=1 的代码形状碰巧和 car_fused 一样。

### 13.5 现在的代码

`新:1634`：

```cpp
{
    // ar_rmsnorm_gfx1250 compiles each square to a rounded multiply and then an
    // add; a contracted fma would change the sum of squares (and its bits)
    // depending on how the compiler schedules the PACKS loops.
#pragma clang fp contract(off)
    ...
```

### 13.6 执行过程

编译期生效，运行时没有任何额外操作。

### 13.7 指令拆解

新版 FP8（`new_fp8_split`）：

```
v_pk_mul_f32 v[14:15], v[12:13], v[12:13]   ; 两个值同时平方，各舍入一次
v_pk_mul_f32 v[16:17], v[10:11], v[10:11]
v_add_f32_e32 v1, v14, v15                   ; 再加，再舍入一次
```

| 指令 | 意思 |
|---|---|
| `v_pk_mul_f32 v[d:d+1], v[a:a+1], v[b:b+1]` | 打包乘法：一条指令算两对 fp32 的乘积，各自舍入 |
| `v_add_f32` | fp32 加法，舍入一次 |
| `v_pk_fma_f32`（被禁止的） | 打包乘加：a × b + c，中间不舍入，只在最后舍入一次。更快也更准，但和参考实现的结果不同 |

### 13.8 通用性和经验

- **凡是要求和参考实现逐位一致的 kernel，都应该显式写 `fp contract(off)`**（或者参考实现也显式写成 fma），不要依赖编译器"碰巧"的选择。
- **逐位比对要覆盖所有代码路径**：每个 PACKS、每种模式的切换点两侧都测，这次的问题就是在 M=300（PACKS 改变的形状）上被发现的。

---

## 14. 方法：这些问题是怎么找到的

单个 tip 能不能用到别处要看场景，下面这些方法每次都用得上。

### 14.1 消融：去掉一个环节，看总时间少多少

**做法**：每次只去掉（或者重复做两遍）kernel 里的一个环节，其余不动，测总时间。变化量就是这个环节的成本。

- 去掉之后结果是错的，没关系，只要收发协议照常工作，测出来的时间就有效。
- 用编译宏控制（例如 `#if LAMPORT_ABL == 1` 跳过规约），每个值编一个版本。

**这次用它得出的结论**：

| 消融 | 结论 | 对应 tip |
|---|---|---|
| 跳过规约 | 规约要 0.77~1.76 us，远超计算量 | #2、#7 |
| 只发送、不轮询不规约 | 一个 CU 光发送就要 6.28 us | #1 |
| 跳过 FP8 的 amax 规约 | 约 0.45 us | #6 |
| 回填做两遍 | 总时间几乎不变，回填很便宜，不值得为它改协议 | 排除了一个方向 |

**价值**：先量出每一段花多少时间，再决定优化哪一段，避免在不重要的地方花力气。

### 14.2 看汇编

**怎么拿到汇编**（这次用的命令）：

```bash
SO=/app/aiter/aiter/jit/module_custom_all_reduce_gfx1250.so
LLVM=/opt/rocm/llvm/bin
$LLVM/llvm-objcopy --dump-section=.hip_fatbin=m.fatbin $SO          # 取出 GPU 代码包
$LLVM/clang-offload-bundler --unbundle --type=o --input=m.fatbin \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 --output=m.co       # 取出 gfx1250 的代码对象
$LLVM/llvm-objdump -d --mcpu=gfx1250 m.co > m.s                     # 反汇编
grep -n '<.*lamport_ar_rmsnorm' m.s                                 # 找到 kernel 的起始行
```

kernel 名字是 C++ 修饰过的，里面能看出模板参数，例如 `IDF16bDF16bLi4ELi1024ELi1ELb0E` 表示 `<bf16, bf16, 4, 1024, 1, false>`（TP4、PACKS=1、row 模式）。

**重点看什么**：

| 搜什么 | 说明什么 | 这次发现的 |
|---|---|---|
| `s_wait_storecnt` | 在等写完成。出现在 barrier 前就要警惕 | #2 |
| `s_load ... s[0:1]` | 读没有预载的 kernel 参数 | #4 |
| `s_wait_kmcnt` 的位置 | 在等参数或标量读，看它是否卡在关键路径上 | #4、#5 |
| `ds_bpermute` | `__shfl` 系列，走 LDS | #7 |
| `s_barrier_signal` 的个数 | 一共几次 block 同步 | #6、#7 |
| `v_pk_fma` / `v_fma` | 有没有乘加合并 | #11 |
| `scratch_` | 寄存器不够或数组被放进了内存 | 确认 PACKS 数组留在寄存器里 |
| `th:` 和 `scope:` | 访存的缓存策略 | #8 |

**读竞争对手的汇编**：Gluon 的 kernel 用同样方法反汇编，对照着看它用了什么指令（#8 的 BYPASS 就是这么找到的）。

### 14.3 kernel 内打时间戳

在 kernel 的每个阶段结束处，让每个 block 的一个线程读一次硬件时钟写进一块调试内存，host 端算出每个阶段的耗时（第 1.4 节的表就是这么得到的）。用编译宏控制，不要用运行时参数控制（运行时参数本身会改变时序，见第 15 节最后一行）。

### 14.4 二分定位回退

性能突然变差、又不知道是哪次修改造成的时候：

1. 找到一个快的版本和一个慢的版本。
2. 把中间的修改一步一步重放出来，每步存一份快照（这次是从会话记录里重放的）。
3. 二分编译、测试快照，找到变慢的那一步。
4. 对比那一步前后的汇编，找到唯一的差别。

#5 就是这么找到的。如果靠猜，很难想到"删一个调试参数"会让 kernel 慢 3 us。

### 14.5 测量要防干扰

- **同一台机器上可能有别人的任务。** 这次有另一个容器间歇地占用 4 张 GPU，导致 main 的 FP8 数据被高估了一倍多（一度算出 2.26 倍加速，实际是 1.33 倍）。
- **做法**：
  - 测之前用 `amd-smi process`、`amd-smi monitor` 看有没有别的进程；
  - 新旧版本**交替**跑多遍（新、旧、新、旧……），每个形状取多遍的中位数；
  - 看同一形状多遍之间的差异，差异大的数据不能用。
- **用 graph 计时**：10 次调用录成一个 graph 反复 replay，去掉 launch 开销，测的是 kernel 本身。

### 14.6 每改一处都做逐位比对

每次修改后都和 car_fused 逐位比对，覆盖：TP2/TP4、bf16/fp16、bf16/FP8 输出、每种模式切换点的两侧、含 NaN 的数据、混合调用序列。#11 的问题就是这样当场发现的。如果等到最后才测，就很难知道是哪一步引入的。

### 14.7 以后优化类似 kernel 的检查清单

按这个顺序过一遍，大部分问题能找到：

1. **先量**：时间戳或消融，找出最大的一两段。
2. **每个 CU 的活够不够均匀**：小规模输入时，是不是只有少数 CU 在干活？发送这类互相独立的工作能不能分散出去？（#1）
3. **同步前在等什么**：barrier 前有没有 `s_wait_storecnt`？barrier 交换的数据是不是只在 LDS？（#2）
4. **参数读**：汇编里有没有 `s_load ... s[0:1]`？有没有读 blockDim/gridDim？结构体/数组参数是不是用了运行时下标？（#4）
5. **规约**：wave 内是不是还在用 `__shfl`？跨 wave 同步了几次？能不能用原子操作（max/min）？（#6、#7）
6. **block 大小**：是不是被问题规模绑死了（一行一个 block，一个线程一个元素）？（#3）
7. **无用访存**：有没有读了又扔的数据？（#9）
8. **访存策略**：缓存策略、地址计算是不是合适？（#8）
9. **按形状选配置**：不同形状最优配置不同时，扫全量、写规则、和最优比。（#10）
10. **逐位一致**：显式控制浮点合并和运算顺序。（#11）

---

## 15. 试过、没用的

| 尝试 | 结果 | 原因 / 教训 |
|---|---|---|
| cluster 的部分和用带返回值的原子交换发布，想避开 storecnt 等待 | 9~35 us，大幅变慢 | 原子操作排在前面的跨卡写后面，还有同一 cache line 的竞争 |
| cluster 里一半 wave 只发送、发完先退出，另一半接收 | 比原版 cluster 更慢 | cluster 规约本身要走一趟显存（写确认 + 作废 cache + 读），约 1.5~3.8 us，省不掉 → 后来改成 split |
| 跨 wave 的串行和用 `readlane` 在 SALU 上做 | 变慢 | CU 上所有 wave 挤在一个 SALU 上 |
| 第一版 split：回填仍由接收块做、规约仍是旧写法 | 比 row 还慢 | 接收块尾部 2304 次回填写 → 改成发送块回填 |
| 发送块大小 = 一整行 | 6144~8192 时 9.5~12.7 us | 又变成一个 CU 发一整行 |
| 发送块只用 32 个包 | 慢 | 发送块和接收块一样大，只用 32 个线程，浪费调度 |
| 申请超过 160 KB 动态 LDS，强制一个 CU 只放一个 block | 没改善 | 说明问题不是 block 被挤到同一个 CU |
| row 模式用 `s_sleep` 代替"等一次参数读" | 最多到 8.1（6144），2048 更差 | 固定时长不能随发送量变化 |
| 读一块设备显存来等待（system scope 的标量读） | M 大时 68~261 us | 不能用 |
| split 接收块也加"等一次参数读" | 变慢 | 接收块不发送，等待没有意义 |
| 在 kernel 开头判断一个调试参数 `dbg` | 每次调用多约 2.5 us | `dbg` 在参数表末尾、不在预载区，kernel 一开始就要等它，挡在发送前面。**插桩、调试开关要用编译宏，不要用运行时参数** |

---

## 16. 术语表

| 术语 | 解释 |
|---|---|
| AR | all-reduce：每张卡各有一份数据，结果是所有卡的数据之和，每张卡都拿到 |
| TP2 / TP4 | 张量并行，2 张 / 4 张卡一起算 |
| car_fused | `ar_rmsnorm_gfx1250`，aiter 里已有的 AR + RMSNorm 实现，本 kernel 要和它逐位一致 |
| Gluon | Triton 的底层方言，PR #8022 用它写了一个对标的 AR + RMSNorm + 量化 kernel |
| 逐位一致（bit-exact） | 输出的每一位都相同，不只是数值接近 |
| IPC 显存 | 一张卡上可以被其他卡直接读写的显存 |
| Lamport 协议 | 不做卡间同步，靠"数据里没有哨兵值"判断数据已到达 |
| 哨兵（sentinel） | 表示"还没写入"的特殊值，这里是 0xffff |
| 回填（rearm） | 把用过的收件箱重新写成哨兵 |
| 轮次计数器（round counter） | 记录每组包当前用哪块缓冲；split 模式拆成 tx（发送块用）、rx（接收块用） |
| 包（pack） | 16 字节，8 个 bf16 |
| row 模式 / split 模式 | 一行一个 block 自己收发 / 发送块和接收块分开，见第 1.5 节 |
| cluster | gfx1250 上几个 block 组成一组，可以做 cluster 级同步；旧版用它把一行拆给多个 block，新版已删 |
| PACKS | 一个线程负责几个包 |
| CU | 计算单元，gfx1250 每卡 256 个 |
| wave / lane | 32 个一起执行的线程 / 其中一个线程 |
| row（DPP 语境） | wave32 的 32 个 lane 分成两行，每行 16 个 |
| block | 若干 wave，在同一个 CU 上，共享 LDS |
| LDS | CU 内的共享内存 |
| VGPR / SGPR | 向量寄存器（每 lane 一份）/ 标量寄存器（每 wave 一份） |
| VALU / SALU | 向量运算单元 / 标量运算单元 |
| exec 掩码 | 记录 wave 里哪些 lane 正在执行 |
| kernarg（kernel 参数区） | host 传给 kernel 的参数，在内存里，地址在 `s[0:1]` |
| 预载（preload） | wave 启动前，硬件把参数区开头的若干 dword 直接放进 SGPR；这里实测 30 个 dword |
| 隐藏参数 | 编译器自动追加在参数区末尾的参数，如 gridDim、blockDim |
| dword | 4 字节 |
| `s_load` / `s_wait_kmcnt` | 标量读 / 等标量读返回 |
| `storecnt` / `loadcnt` / `dscnt` | 未完成的显存写 / 显存读 / LDS 操作的计数，`s_wait_*` 等它们降到指定值 |
| `xcnt` | 写指令的数据还没离开寄存器的计数；等它是为了能改写那些寄存器 |
| barrier（block 同步） | block 内所有 wave 都到这里才继续，gfx1250 上是 `s_barrier_signal` + `s_barrier_wait` |
| fence（栅栏） | 规定内存操作的可见顺序：release（之前的写先可见）/ acquire（之后能看到别人的写） |
| scope | 可见范围：workgroup（block 内）、cluster、SE、device、system（跨卡） |
| 蝴蝶规约 | wave 内求和：5 轮，每轮和 lane ^ 16/8/4/2/1 交换相加 |
| DPP | 附在 VALU 指令上的修饰，让一个操作数直接取自别的 lane |
| `row_xmask:N` | DPP 规则：读同一行内 lane ^ N |
| `bound_ctrl` | DPP 选项：源 lane 无效时读 0 还是保留旧值 |
| `permlanex16` | 跨行取值的指令，这里用来实现 lane ^ 16 |
| `ds_bpermute` | 借 LDS 交换电路在 lane 之间取值，`__shfl` 系列编译成它 |
| VOPD（`v_dual_*`） | 两条互不依赖的运算打包成一条指令 |
| buffer 指令 / 描述符（V#） | 通过 4 个 SGPR 的描述符（基址、上限、格式）加 32 位偏移访问内存的指令 |
| TH（temporal hint） | 缓存策略，BYPASS 表示不在缓存里保留 |
| FMA / 乘加合并 | a × b + c 一条指令完成，只舍入一次；`fp contract` 控制编译器能否自动合成 |
| amax | 一行输出的绝对值最大值，用来算 FP8 量化的 scale |
| 消融（ablation） | 去掉一个环节测总时间的变化，得到这个环节的成本 |
| graph replay | 把多次 kernel 调用录成一个 HIP graph 反复执行，用来精确计时 |
