# Lamport AR + RMSNorm kernel 逐段讲解

> - 文件：`/app/aiter/csrc/include/custom_all_reduce_gfx1250.cuh`，分支 `gfx1250_lamport_rms_opt`，提交 `abb37852a`
> - 行号以这个提交为准
> - 标 **【临时】** 的代码是实验用的，提 PR 前会删掉

---

## 0. 先看全局

### 0.1 这个 kernel 做什么

每张卡有一份 `input[rows][hidden]`（bf16）。kernel 一次完成：

```
sum      = 4 张卡的 input 按卡号顺序相加（fp32 累加，再舍入成 bf16）
res_out  = sum + residual_inp          → 写 residual_out
out      = res_out * rsqrt(mean(res_out^2) + eps) * weight
           bf16 输出：直接写 output
           FP8 输出：再按行求 amax，量化成 FP8，写 output 和 scale_out
```

结果和 `ar_rmsnorm_gfx1250`（测试里的 car_fused）逐位一致，所以每一步的加法顺序都照抄它。

### 0.2 几个基本单位

| 名词 | 含义 |
|---|---|
| 包（pack） | 16 字节 = 8 个 bf16。一次读写就是一个包 |
| row_packs | 一行有多少个包 = hidden / 8，例如 6144 → 768 |
| wave | 32 个线程一起执行（gfx1250 的 warp） |
| 一组包（group） | 连续 32 个包，正好一个 wave 处理；每组有自己的轮次计数器 |
| block | 一个 CU 上运行的一组 wave |

### 0.3 收件箱：数据怎么在卡之间走

每张卡开一块显存当收件箱，通过 IPC 让另外 3 张卡能直接写进来：

```
每张卡的收件箱 inbox[轮次缓冲 0..2][发送方 rank 0..3][包 0..slot-1]，每格 16 字节
初始化时整块填成 0xff
```

第 r 轮，卡 A 把自己的第 p 个包，写到每张别的卡 B 的 `inbox[r][A][p]`。
卡 B 读自己的 `inbox[r][A][p]`，只要还有 0xffff 就说明还没到，继续读（轮询）。

**为什么要 3 块缓冲轮换**：最快的卡最多比最慢的卡领先一轮。第 r 轮用缓冲 r%3，同时把缓冲 (r+2)%3（也就是上一轮用过的那块）重新填成 0xffff，叫 **回填（re-arm）**，留给下下轮用。

**轮次计数器**：每组 32 个包有一个计数器，记录这组包下次用哪块缓冲（0、1、2 循环）。所有卡执行相同的调用序列，所以同一组包的计数器在每张卡上都一样，不需要卡之间同步。

### 0.4 三种运行模式

```
row 模式：每行一个 block，同一个 block 既发送又接收
  block r: 读本卡数据 → 发给 3 张卡 → 等一次参数读 → 读 residual/weight → 轮询 → 计算 → 回填、更新轮次

split 模式：发送和接收分给不同的 block
  grid = [发送块 0 .. senders-1][接收块：每行一个]
  发送块: 读轮次(tx) → 读本卡数据 → 发给 3 张卡 → 回填 → tx+1 → 退出
  接收块: 读轮次(rx) → 读本卡数据、residual、weight → 轮询 → 计算 → rx+1

cluster 模式：一行拆给几个 block，组成一个 workgroup cluster，跨 block 求和
  （小 M 时很慢，准备删掉）
```

现在测下来最好的组合：M ≤ 64 用 split（接收块每线程 1 个包、发送块 256 线程），M ≥ 128 用 row（每线程 2~4 个包）。自动选择逻辑还没写，目前要靠环境变量指定。

---

## 1. 常量（1387–1395）

```cpp
1387 // #define LAMPORT_STAMPS 1 // TEMP        【临时】打开后在 kernel 里记时间戳
1388 kLamportBuffers = 3                         轮换的收件箱缓冲数，见 0.3
1389 kLamportWave = 32                           一个 wave 的线程数
1390 kLamportRowMaxThreads = 1024                一行最多 1024 个包 → hidden ≤ 8192
1391 kLamportClusterMaxThreads = 256             cluster 模式每块最多 256 线程
1392 kLamportMaxClusterBlocks = 16               一个 cluster 最多 16 块
1395 kLamportMinClusterRowPacks = 256            一行 ≤ 256 包时不拆 cluster
```

1397–1433：判断编译器和 HIP 头文件支不支持 cluster，不支持就把 cluster 路径编译掉。和主流程无关。

## 2. 收件箱大小（1435–1468）

```
lamport_ar_rms_buffer_bytes = 3 × ngpus × slot × 16     数据收件箱（本 kernel 用的就是这块）
lamport_escape_bytes        = 3 × ngpus × slot × 4      all-gather 用的附加区
lamport_two_shot_bytes      = ngpus × slot × 16         2-shot all-reduce 用的附加区
```

slot 是一次调用最多能有多少个包（TP4 默认 8 MiB / 16 = 524288）。后两块是别的算子用的，同一块显存连在后面。

## 3. 哨兵 0xffff（1475–1484）

```cpp
1476 uint32_t lamportSentinelMask(uint32_t w)
       x   = ~w
       tmp = ((x & 0x7fff7fff) + 0x7fff7fff) | x
       return (tmp & 0x80008000) ^ 0x80008000
```

一个 uint32 里有两个 16 位。哪个 16 位正好等于 0xffff，就把那个 16 位的最高位置 1，其他位都是 0。结果不等于 0 就表示"还有没到的数据"。

```cpp
1484 lamportCanonicalize(w) = w ^ lamportSentinelMask(w)
```

把数据里恰好是 0xffff 的 16 位改成 0x7fff。0xffff 和 0x7fff 在 bf16 里都是 NaN，改了不影响数值意义，但能保证真实数据永远不会被误认成"还没到"。发送前每个包都要过一遍。

## 4. 读写函数（1486–1535）

```cpp
1486 lamport_u32x4                  16 字节看成 4 个 uint32
1488 lamport_store_b128 / 1497 lamport_load_b128
                                    普通的 16 字节写/读（scope:SCOPE_SYS），1-shot、2-shot 等别的 kernel 在用
1506 LAMPORT_POLL_MAXIT (1 << 24)   【临时】轮询次数上限，防止协议出错时把 GPU 卡死
1507 LAMPORT_ABL 0                  【临时】消融实验开关，0 = 关
1508 LAMPORT_BYPASS 1               【临时】开关，正式版直接保留 BYPASS 写法
```

```cpp
1514 lamport_store_sys(base, pack, v)
       __builtin_amdgcn_raw_buffer_store_b128(v,
           make_buffer_rsrc(base, 0, 0x7fffffff, 0),   ← 以 base 为起点生成 buffer 描述符
           (int)(pack * 16),                           ← 字节偏移
           0, 27)                                      ← cache 策略：TH_BYPASS + SCOPE_SYS
1524 lamport_load_sys(base, pack)                      对应的读
```

本 kernel 的跨卡发送、轮询、回填都用这两个函数。`27` 让指令带上 `th:TH_*_BYPASS scope:SCOPE_SYS`，和 Gluon 一样：SYS 保证跨卡可见，BYPASS 提示 cache 不要保留这些只用一次的数据。

## 5. 同步和等待（1537–1559）

```cpp
1540 lamport_lds_barrier()
       fence(release, workgroup, local)    只等本 wave 之前的 LDS 写完成
       s_barrier                           block 内所有 wave 到齐
       fence(acquire, workgroup, local)
```

block 内同步，但**只保证 LDS（共享内存）的数据可见**。`__syncthreads()` 还会等所有显存写入完成，包括刚发出的跨卡写，要多等一个来回（约 1 us）。这里同步只是为了交换 LDS 里的部分和，所以用这个轻量版。

```cpp
1552 lamport_kernarg_wait()
       args = kernel 参数区的地址
       v = args[128 / 4]             读偏移 128 处的参数（不在预载范围内，要真的去显存读）
       asm volatile("" :: "s"(v))    强制等这个值返回
```

只在 row 模式发送之后调用一次：等这一次参数读返回之后，才去读 residual/weight 并开始轮询。实测能让 row 模式快很多（6144×1：10.4 → 7.6 us）；用固定时长的 sleep 代替没有这个效果。原因没有证实，推测是这次读要排在刚发出的跨卡写后面才能返回，相当于"等写发出去了再轮询"。

## 6. 规约（1561–1620）

```cpp
1565 lamport_xor_lane<OFF>(v)       取 lane (i xor OFF) 的值，全在寄存器里完成
       OFF = 16：permlanex16（32 个 lane 分两排，每个 lane 换到另一排的同一位置）
       OFF = 1/2/4/8：DPP row_xmask（16 个 lane 一排，排内 i xor OFF）
```

```cpp
1579 lamport_wave_allreduce<kMax>(p)
       p = p + xor16(p); p = p + xor8(p); ... xor4, xor2, xor1
```

一个 wave 内 32 个 lane 两两相加 5 轮，每个 lane 都得到 32 个数的和（kMax 时求最大值）。加法顺序和 car_fused 的 `block_allreduce_f32` 完全一样，所以结果逐位一致。

```cpp
1599 lamport_row_allreduce<kMax, PACKS>(v[PACKS], row_groups, waves)
1601   __shared__ float partial[32]           一行最多 32 组（8192 / 8 / 32）
1604   for k:                                 每线程负责 PACKS 个包，各有一个部分和
1606     p = 本 wave 求和(v[k])
1607     lane 0 把它写到 partial[第几个 wave + k × waves]   ← 正好就是这一组在行里的编号
1610   lamport_lds_barrier()                  整个 block 同步一次
1612   求最大值：每个 lane 读一个 partial，再做一次 wave 内最大值
1616   求和：每个 lane 从 partial[0] 开始按顺序加到 partial[row_groups-1]
```

求和必须按组的编号顺序加（照抄 car_fused），所以每个 wave 自己从头加一遍，不再同步第二次。求和和求最大值是两个不同的模板实例，各有一份 `partial`，互不覆盖。

## 7. kernel 参数（1637–1667）

```cpp
template <T, OutT, ngpus, MAX_THREADS, CLUSTER, PACKS, SPLIT>
  T           bf16 / fp16
  OutT        等于 T 是 bf16 输出；FP8 类型是 FP8 输出
  ngpus       卡数，编译期常量
  MAX_THREADS block 最大线程数（launch bounds）
  CLUSTER     是否 cluster 模式
  PACKS       row 模式每线程处理几个包
  SPLIT       是否 split 模式
```

参数顺序是故意排的：**只有前 30 个 dword（前 120 字节）会在 wave 启动时直接放进 SGPR（预载）**，后面的参数用到时要现从显存读，很慢。所以发送和 epilogue 要用的参数都放在前 120 字节：

| 字节偏移 | 参数 | 说明 |
|---|---|---|
| 0–47 | input, residual_inp, residual_out, output, weight, round_counter | 6 个指针 |
| 48 | slot | 每个发送方在收件箱里占多少个包 |
| 52 | rank_gemma | 低 8 位是 rank，第 8 位是 gemma_norm（预载区放不下单独一个参数） |
| 56 | hidden | |
| 60 | senders | split 模式发送块的个数 |
| 64–95 | p0–p3 | 4 张卡收件箱的地址 |
| 96–111 | scale_out, bf16_output | FP8 输出用 |
| 112 | eps | |
| 116 | send_packs | split 模式每个发送块发多少个包 |
| 120 以后（不预载） | p4–p7, cluster_partials | 只有 6/8 卡或 cluster 模式才用 |

## 8. kernel 主体

### 8.1 开头准备（1669–1700）

```cpp
1672 #pragma clang fp contract(off)
```
禁止编译器把"乘再加"合并成一条 fma（fma 中间不舍入，平方和的最低位会和 car_fused 不同）。

```cpp
1676 rank = rank_gemma & 0xff;  gemma_norm = 第 8 位
1678 QUANT = OutT 不是 T（即 FP8 输出）
1679 pack_size = 8，P = 8 个 bf16 的向量
1681 union pack { P data; lamport_u32x4 i; }   同 16 字节，算数时当 8 个 bf16，比较/收发时当 4 个 uint32
```

```cpp
1689 all[8] = {p0, ..., p7}
1690 own = all[rank]（用编译期展开的循环选出来）
```
所有下标都是编译期常量。如果写成 `all[rank]` 这种运行时下标，每次取地址都要先发一条标量读，会挡在发送前面。

```cpp
1699 row_packs  = hidden / 8
1700 rx_counter = round_counter + slot/32      计数器数组的后半段是 rx，前半段是 tx
```

**tx / rx**：split 模式里发送块和接收块是不同的 block，没法保证谁先读计数器、谁先改。所以分成两份：发送块只读写 tx，接收块只读写 rx。不分模式的 kernel（row、1-shot、2-shot、AG）两份都写，保证各种算子交替调用时两份始终相等。

### 8.2 split 的发送块（1701–1737）

```cpp
1703 if (blockIdx.x < senders)                       前 senders 个 block 是发送块
1705   t = 本 wave 第一个线程在 block 里的序号
1707   if (t >= send_packs) return                    发送块和接收块线程数一样，多出来的 wave 直接退出
1709   p     = blockIdx.x * send_packs + threadIdx.x  本线程负责第 p 个包（全局编号）
1710   wave  = p / 32                                 属于哪一组
1711   round = round_counter[wave]                    读 tx 轮次
1713   v = input 的第 p 个包；1716 canonicalize
1717   sent = (round × ngpus + rank) × slot + p       对方收件箱里的位置：inbox[round][我][p]
1719   对每张别的卡 g：写到 all[g] + sent             跨卡发送
1725-1732 回填：把自己收件箱 inbox[(round+2)%3][每个别的卡][p] 写成 0xffff
          （1728 那层 rep 循环是 【临时】消融开关，正常就是写一次）
1733   lane 0：round_counter[wave] = round + 1        tx 轮次加一
1735   return
```

回填为什么可以由发送块来做：缓冲 (round+2)%3 是上一轮用过的，上一轮的 kernel 已经结束，数据已经被读过了；对方最多领先一轮，这一轮里不会往这块写。

### 8.3 确定本线程负责哪些包（1738–1760）

```cpp
cluster:  block = blockDim.x；tid0 = 全局线程号；row = tid0 / row_packs
row/split: block = row_packs / PACKS（等于 blockDim.x，但不用去读参数区）
           row  = blockIdx.x（split 要减去发送块的个数）
           tid0 = row × row_packs + threadIdx.x

1756 tid[k]   = tid0 + k × block      本线程负责第 tid0、tid0+block、tid0+2×block … 个包
1759 round[k] = 这个包所在组的轮次（split 读 rx，其他读 tx）
```

例：6144 一行 768 个包，PACKS=1 时 block 有 768 线程，线程 t 负责第 t 个包；PACKS=2 时 384 线程，线程 t 负责第 t 和 t+384 个包。

### 8.4 读本卡数据并发送（1762–1785）

```cpp
1766 local[k] = input 的第 tid[k] 个包；1769 canonicalize
1772 if (!SPLIT)                       row/cluster：自己发送
1777   sent = (round[k] × ngpus + rank) × slot + tid[k]
1779   对每张别的卡写一次
1784 if (!SPLIT && !CLUSTER) lamport_kernarg_wait()    row 模式：发完等一次参数读（见第 5 节）
```

split 的接收块也要读 `local`：求和时本卡这一份直接用自己的输入，不经过收件箱。

### 8.5 先读 residual 和 weight（1790–1802）

```cpp
1799 res[k] = residual_inp 的第 tid[k] 个包
1800 w[k]   = weight 的第 (tid[k] - row × row_packs) 个包，即本行内的列号
```

这两个都在本卡，提前发出读请求，等待数据到达的同时它们也在路上，不额外占时间。（1795 是【临时】消融开关。）

### 8.6 轮询（1804–1841）

```cpp
1806 scratch[PACKS][ngpus-1]           存 3 张别的卡发来的包
1811 do {
1814   asm volatile("" ::: "memory")   告诉编译器内存可能变了，不许把读挪到循环外面
1818   inbox = round[k] × ngpus × slot + tid[k]
1822   g = rank+1, rank+2, rank+3（模 4）：只读别的卡那几格，自己那格永远是 0xffff
1823   scratch[k][i] = 读 own 收件箱的 inbox[round][g][tid[k]]
1827   sched_barrier(0)                 先把这一轮的读全部发出去，再开始检查
1835   m |= 每个 uint32 的哨兵掩码
1836   pending = (m != 0)               还有 0xffff，说明有数据没到
1838   【临时】超过上限就退出
1841 } while (pending)
```

注意这是**每个线程各自轮询自己的包**，不需要整个 block 一起等。一个 wave 内只要还有一个 lane 没收齐，整个 wave 就继续循环。

### 8.7 求和、加 residual、平方和（1846–1879）

```cpp
1857 for g = 0 .. ngpus-1:                    按卡号 0、1、2、3 的顺序加（照抄 car_fused）
1859   at = (g - rank - 1 + ngpus) % ngpus    卡 g 的数据在 scratch 的第几格；at = 3 表示是本卡
1860-1863 v = 本卡就用 local，否则用 scratch[k][at]
1866   x[k][j] += float(v[j])                 fp32 累加
1873 x = float(bf16(x)) + float(res)          all-reduce 结果先舍入成 bf16，再加 residual（fp32）
1874 sum = bf16(x)                            → residual_out
1875 square_sum[k] += x × x                   本包 8 个值的平方和
1878 写 residual_out（1877 的条件是【临时】消融开关）
```

### 8.8 整行求 inv_rms（1881–1915）

```cpp
1881 row_groups = row_packs / 32                 一行有几组
1883-1909 cluster：部分和写到显存，cluster 同步后读回来相加（准备删掉）
1912-1914 row/split：
       total   = lamport_row_allreduce<求和>(square_sum, row_groups, block/32)   见第 6 节
       inv_rms = rsqrt(total / hidden + eps)
       （1912 的 LAMPORT_ABL==1 分支是【临时】消融开关）
```

### 8.9 乘 weight，写 bf16 输出（1920–1936）

```cpp
1927 wj = float(w) + (gemma_norm ? 1 : 0)      gemma 版 RMSNorm 用 (1 + weight)
1928 o  = bf16(x × (inv_rms × wj))             先算 inv_rms × wj，再乘 x（顺序照抄 car_fused）
1935 bf16 输出：直接写 output
```

### 8.10 FP8 输出（1937–2004）

```cpp
1950 amax[k] = max |o|                          本线程 8 个值（已经舍入成 bf16 的输出）的最大绝对值
1951 如果 bf16_output 不为空，同时写一份 bf16 输出
1982 row_amax = lamport_row_allreduce<求最大值>(amax, ...)     整行最大值，block 第二次同步
1984 fp8_max = FP8 能表示的最大值（448）
1985 scale   = row_amax / fp8_max
1986 inv     = 1 / scale（硬件近似倒数指令 rcp，和 car_fused 一样）
1994 a = clamp(o × inv, -fp8_max, fp8_max)       med3 一条指令完成 clamp
1997 两个 float 打包转成两个 FP8
2000 写 output（8 个 FP8 = 8 字节）
2002 每行第一个线程写 scale_out[row]
```

FP8 比 bf16 多的成本主要就在 1982 这次整行规约：要等整行的输出都算完，block 再同步一次。消融测下来约 0.45 us，这是第 3 步要优化的地方。

### 8.11 回填和更新轮次（2006–2027）

```cpp
2008 row/cluster（split 已由发送块回填）：
2010   rearm = (round[k] + 2) % 3
2014   自己收件箱 inbox[rearm][每个别的卡][tid[k]] 写成 0xffff
2017 每个 wave 的 lane 0：
2024   row/cluster：tx 轮次 + 1
2025   rx 轮次 + 1
```

### 8.12 时间戳（2028–2038）

【临时】`LAMPORT_STAMPS` 打开时，把各阶段时间写到 residual_out 里，用脚本读出来。默认关闭。

---

## 9. host 端：`LamportArRms::run()`（2665–2780）

```cpp
2679 检查：hidden 必须是 256 的倍数且 ≤ 8192；总包数 packs = rows × row_packs 不能超过 slot
2693 threads = cluster_threads(rows, row_packs)
       行数少于 CU 数、一行超过 256 个包时，返回 cluster 每块的线程数，否则 0（row 模式）
```

```cpp
2695-2704 【临时】环境变量开关，每次调用都重新读：
       AITER_LAMPORT_RMS_CLUSTER    0 强制不用 cluster，1 强制 cluster
       AITER_LAMPORT_RMS_PACKS      每线程几个包
       AITER_LAMPORT_RMS_CBS        cluster 每块线程数
       AITER_LAMPORT_RMS_SENDPACKS  split 发送块大小（默认 64）
       AITER_LAMPORT_RMS_LDS        额外申请的 LDS（试过，没用）
       AITER_LAMPORT_RMS_SPLIT      1 = split 模式
2706-2713 根据开关决定 cluster 还是 row/split
2714-2726 决定每线程几个包：
       环境变量指定了就用指定的
       split 且没指定时：选最小的 PACKS 让接收块不超过 8 个 wave
       （这个默认值不是最优：实测 PACKS=1 最快，自动选择时要改）
2727 row_threads = row_packs / PACKS          接收块（或 row 模式 block）的线程数
2728-2730 send_packs：32 的倍数，并且能整除 row_packs，保证最后一个发送块不会越界
2731 senders = packs / send_packs             发送块个数
```

```cpp
2733 LAMPORT_RMS_ARGS   按第 7 节的顺序传参，rank 和 gemma_norm 合成一个 int
2739 LAUNCH_LAMPORT_RMS_ROW(NG, PK)
       split：grid = senders + rows 个 block，每块 row_threads 线程
       row：  grid = rows 个 block，每块 row_threads 线程
2748 LAUNCH_LAMPORT_RMS(NG)
       threads == 0：按 PACKS 选 1/2/3/4/8 对应的模板实例
       否则 cluster：hipLaunchKernelEx 带 cluster 维度启动；
           2774 再补一句，因为 HIP graph 录制时会丢掉 cluster 维度（以前踩过的坑）
2776 DISPATCH_NGPUS_1250：按卡数 2/4/6/8 选模板实例
```

**现在不设环境变量时的默认行为**：小 M、大 hidden 走 cluster（很慢），其他走 row、每线程 1 个包。实测最好的组合要靠环境变量打开，自动选择逻辑还没写。

---

## 10. 提 PR 前要删或要改的

| 位置 | 内容 |
|---|---|
| 1387、1694、1787、1843、1917、2028 | `LAMPORT_STAMPS` 时间戳 |
| 1506、1808、1837 | `LAMPORT_POLL_MAXIT` 调试上限 |
| 1507、1728、1795、1877、1912、1982、2008 | `LAMPORT_ABL` 消融开关 |
| 1508 | `LAMPORT_BYPASS` 开关（保留 BYPASS 写法，删开关） |
| 2695–2704 | 环境变量开关，换成自动选择逻辑 |
| 1652 | 注释"a bool argument is not preloaded"是错的，改成"预载区放不下" |
| cluster 相关（1739–1744、1883–1909、1955–1979、2760–2775 等） | 整条 cluster 路径删掉 |
