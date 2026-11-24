# gfx1250 Lamport AR + RMSNorm：对标 PR #8022 Gluon 的优化记录

> - **整理日期**：2026-11-18
> - **机器**：4 × gfx1250（MI455X 类），每卡 256 CU，ROCm 10.1，TP4 走 IPC
> - **代码位置**：`/app/aiter`，本地分支 `lamport_rms_opt`，基于 PR #8022 的 head `1aef2c832`（lirui927，分支 `fuse/ar-norm-fp8`）
> - **存档**：本地 WIP commit `c85362407`（**没有 push**）；同目录 `pr8022_wip/` 里有一份 patch、测试脚本和关键日志
> - **涉及文件**：只改了 `csrc/include/custom_all_reduce_gfx1250.cuh`
> - **相关文档**：同目录 `fused_poc_notes.md`（gfx942 上的融合 POC）、`split_h_and_cu_vmem_limits.md`（切分 H 与 CU 访存上限）

## 0. 下次接着做：先看这里

### 0.1 目标

PR #8022 加了一个 Gluon 写的 `all_reduce_rmsnorm_per_token_quant`，只在 TP4、`AITER_GLUON_FUSED_AR=1` 时生效，M ≤ 128 的调用会交给它。任务是找出 Gluon 比我们的 HIP Lamport kernel（`lamport_ar_rmsnorm_gfx1250`）快的形状，优化 Lamport，在这些形状上反超。

### 0.2 当前状态

- **已经赢的**：hidden 2048 的所有 M；4096、8192 在 M ≥ 32~64 时也赢。M=96~192 原来那个台阶（第 1 节）也消掉了。
- **还在输的**：hidden 6144 的小 M，6144×1 是 7.24 对 Gluon 6.14，差约 1.1 us。6144 是 Gluon 最擅长的尺寸，一个 wave 正好处理 768 个值。
- **还没测**：5120、7168 在最新改动之后没重测；TP2 还没跑回归。
- **代码状态**：各种变体靠临时环境变量手动选择，还没写成自动 dispatch。代码里还留着失败的实验路径，见 0.4。
- **测量可信度**：最后一批测量期间，GPU 1 被别的容器占用（PID 1594258，显存 4.8 GB，GFX 100%）。split 的几组数（第 6 节）和"只读 3 个 peer"那组（第 7 节）可能受了影响，**必须重测**。

### 0.3 恢复工作的步骤

```bash
cd /app/aiter
git checkout lamport_rms_opt            # 存档 commit c85362407
# 如果分支丢了：git checkout -b lamport_rms_opt 1aef2c832 && git am /app/my_doc/lamport_ar_rms/pr8022_wip/0001-*.patch
amd-smi monitor -u; amd-smi process     # 先确认 4 张卡都空闲，没有别人的进程
df -h /                                 # 先确认磁盘还有空间（11-18 时根分区 100% 满）

# 改了 .cuh 一定要删 .so 再编，否则 JIT 不会重编
rm -rf aiter/jit/module_custom_all_reduce_gfx1250.so aiter/jit/build/module_custom_all_reduce_gfx1250
S=/app/my_doc/lamport_ar_rms/pr8022_wip/scripts
PYTHONPATH=/tmp/agfull/noflydsl:/app/aiter python $S/build.py
```

`/tmp/agfull/noflydsl` 用来屏蔽 flydsl 版本检查。如果 /tmp 被清了，从 `/root/.claude/projects/-app-aiter/notes/noflydsl` 拷回来。

### 0.4 待办（按顺序）

1. **在干净的机器上重测全部尺寸。** TP4，hidden 2048/4096/5120/6144/7168/8192，M 1~256，bf16 和 FP8 都测，变体 row1 / row2 / row4 / split(send 32/64/128)。
2. **写自动 dispatch。** 按测量结果，在 `LamportArRms::run`（约第 2194 行 `TEMP tuning knobs`）里写选择规则：
   - 小 M、大 hidden 走 split；
   - 其余走 row，每线程包数（PACKS）按 hidden 选；
   - 删掉环境变量开关。
3. **删掉失败的实验。**
   - cluster 里 sender wave 先退出的写法（第 1474 行的 `sender`）比原版 cluster 慢，回退到原版，或者如果 row/split 已经全面更快，就整个删掉 cluster 路径。
   - 删掉 `LAMPORT_STAMPS` 时间戳宏（第 1141 行起）。
4. **跑完整正确性测试。**
   - `op_tests/multigpu_tests/test_fused_ar_rms_lamport.py -t 2 4`：和 car_fused 逐位比对，以及 mixed sequence。
   - `test_ar_add_lamport.py`、`test_ag_lamport.py`：round counter 的布局改了，AR+add、AG 现在也要写 rx 计数器（第 6 节），必须验证三者交替调用不出错。
   - TP2 全尺寸回归：PR 作者提过，peer 指针改成标量参数的版本在 TP2、≥256 行、hidden ≥7168 时慢过 10~20%，要确认我们没有这个问题。
5. **继续追 6144 小 M。** 线索见第 9 节。

---

## 1. 基线：Gluon 在哪些形状上更快

测量方法：每次 10 个调用捕获成一个 HIP graph，replay 20 次取中位数，取 4 个 rank 里最慢的那个，单位 us。脚本是 `scripts/bench.py`，日志是 `logs/sweep_base.log`（PR head 原版代码，各跑 2 轮）。

bf16 输出（不 quant），TP4：

| hidden | M=1 Lamport / Gluon | M=8 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|
| 2048 | 7.05 / 6.09 | 7.71 / 6.55 | 8.26 / 7.70 | 8.68 / 8.61 | 10.02 / 11.51 |
| 4096 | 7.77 / 6.55 | 8.20 / 7.13 | 9.03 / 8.80 | 10.39 / 11.53 | 14.49 / 15.75 |
| 6144 | 8.27 / 6.14 | 8.78 / 6.87 | 9.89 / 9.02 | 14.18 / 12.32 | 17.87 / 17.01 |
| 8192 | 8.59 / 7.79 | 8.85 / 8.42 | 10.94 / 11.39 | 15.95 / 17.48 | 21.34 / 23.03 |

- Gluon 在 M ≤ 64 基本全赢，6144 一直赢到 M=256。
- FP8 输出时 Lamport 每次要多付约 1 us，而 Gluon 几乎不多花时间。最差的 6144×1 是 9.26 对 6.35。
- Lamport 有个台阶，比如 6144 从 M=64 到 M=96 是 9.89 → 13.26。原因是 cluster 模式下，block 总数（行数 × 每行 block 数）超过了 256 个 CU。

**Gluon 的结构**（`aiter/ops/triton/_gluon_kernels/gfx1250/comms/all_reduce.py`）：

- 一个 16 字节的"行"里装 3 个 bf16 和一个 16 位的轮次标签。靠标签判断数据是否到达，所以不需要把收件箱重新填成 0xffff（rearm）。
- 发送程序和接收程序分开：前面若干程序只负责发，后面每行一个程序（8 个 wave）只负责收、做 RMSNorm、写输出。

---

## 2. 一行的平方和是怎么算的（后面几节都要用）

以 hidden=2048、bf16、一行一个 block 为例。一个 block 有 256 个线程，分成 8 个 wave，每个 wave 32 个线程（lane 0~31）。每个线程负责一个 16 字节包，也就是 8 个 bf16，256 × 8 = 2048 正好一行。

每个线程依次做这些事：

```
① 读本卡输入 → 写到另外 3 张卡收件箱的对应位置      （跨卡写，global store）
② 轮询读自己的收件箱，直到 3 张卡的数据都不是 0xffff  （global load，数据进寄存器）
③ 4 份按卡号顺序相加 → 舍入成 bf16 → 加残差 → 得到 8 个 fp32 值 x[0..7]
   写 residual_out
   square_sum = x0*x0 + x1*x1 + ... + x7*x7   （每个平方先舍入，再加）
④ 整个 block 把 256 个 square_sum 加成一行的总和 r
   4a. 每个 wave 内：32 个 lane 用"蝴蝶"方式两两相加 5 轮，每个 lane 都得到本 wave 的和
   4b. 每个 wave 的 lane 0 把本 wave 的和写进 LDS：partial[w]
   4c. block 同步
   4d. 按 partial[0] + partial[1] + ... + partial[7] 的顺序加出 r
⑤ inv_rms = rsqrtf(r / hidden + eps)，乘权重，写输出（FP8 时还要再做一次求最大值）
⑥ 把上上轮的收件箱重新填成 0xffff（rearm），轮次计数器加 1
```

4a 的蝴蝶求和：

```
第 1 轮：lane i 拿 lane (i xor 16) 的数，加到自己身上
         lane 0 = s0+s16，lane 16 = s16+s0（两者相同）... → 每个 lane 持有 2 个数的和
第 2 轮：拿 lane (i xor 8) 的数相加 → 4 个数的和
第 3 轮：xor 4 → 8 个数的和
第 4 轮：xor 2 → 16 个数的和
第 5 轮：xor 1 → 32 个数的和，32 个 lane 手里都是同一个值
```

**加法的顺序不能随便改。** 浮点加法换个顺序，最低位可能不同。car_fused（`ar_rmsnorm_gfx1250` 里的 `block_allreduce_f32`）用的就是"wave 内 xor 16/8/4/2/1，跨 wave 按编号顺序串行加"。我们的输出要和它逐位一致，测试里 `lamport bitdiff` 必须是 0，所以下面所有改动都保持这个顺序。

---

## 3. 优化一（效果最大）：block 同步时不再等跨卡写的确认

**代码**：第 1247 行 `lamport_lds_barrier()`，在第 1290 行 `lamport_row_allreduce()` 里调用。

### 3.1 问题

GPU 执行写指令时只是把写请求交出去，线程立刻往下走。每个 wave 有一个计数器 `storecnt`：交出一个写加 1，收到内存那头"写好了"的确认减 1。`s_wait_storecnt 0` 的意思是停下来，直到之前所有写的确认都回来。

普通的写后面编译器**不会**插这个等待，你的 AR+add kernel 发完跨卡写就直接开始轮询，ISA 里看得到。它只在 release 语义的同步处插，因为语义要求"我之前写的东西，别人在同步之后能看到"。

第 ④ 步原来用 `__syncthreads()`，编译出来是：

```
s_wait_storecnt_dscnt 0x0     ← 等所有写的确认 + 所有 LDS 操作
s_barrier_signal -1
s_barrier_wait
```

这里等的"写"包括第 ① 步发往另外 3 张卡的跨卡写（还有第 ③ 步的 residual_out）。跨卡写的确认要走一个来回：

```
时间 0   : 我写给对方，对方也写给我（4 张卡几乎同时）
时间 ≈ L : 对方的数据到了我的收件箱 → 我第 ② 步轮询成功
           同时我的数据也到了对方，但确认才刚开始往回走
时间 ≈ 2L: 我那 3 次跨卡写的确认才回到我这里
```

所以第 ② 步刚结束时，我自己的跨卡写还没"完成"，`__syncthreads()` 要白等约一个单程 L（实测约 1 us）。

### 3.2 改法

第 ④ 步 block 内交换的只有 LDS 里的 `partial[]`，同步之后也没有线程去读别人写的显存。所以只需要保证 LDS 的写对别人可见：

```cpp
DINLINE void lamport_lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); // 只管 LDS
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
```

编译后只剩 `s_wait_dscnt 0x0`（等 LDS，几十个时钟）。跨卡写的确认挪到 kernel 结束时才等，和第 ⑤ 步的计算、写输出重叠了。

### 3.3 适用条件

**只适用于：block 同步只为 LDS 里的数据服务，并且同步之后没有线程读同 block 其他线程刚写的显存。** 如果通过显存交换数据，必须保留完整的 `__syncthreads()`。

Lamport 协议本身不依赖这个同步：数据是否到达，靠收件箱里有没有 0xffff 判断。

### 3.4 效果

消融测试（跳过第 ④ 步作对照）：原来规约要花 0.77 us（2048，row 模式）到 1.76 us（6144，cluster 模式）。改完后 2048×1 从 7.05 降到约 5.5 us，第一次反超 Gluon（6.09）。

cluster 模式（一行拆成多个 block 组成 cluster）用的是 `fence(release, "cluster")` 加 cluster 同步。部分和必须经过显存交换，这个等待去不掉，见第 8 节的失败尝试。

---

## 4. 优化二：wave 内求和改用寄存器直接交换，跨 wave 只同步一次

**代码**：第 1258 行 `lamport_xor_lane`，第 1272 行 `lamport_wave_allreduce`，第 1290 行 `lamport_row_allreduce`。

### 4.1 wave 内（第 4a 步）

每一轮"拿 lane (i xor N) 的数"只是取数，加法在外面单独做：

```cpp
p = p + lamport_xor_lane<16>(p);
p = p + lamport_xor_lane<8>(p);
p = p + lamport_xor_lane<4>(p);
p = p + lamport_xor_lane<2>(p);
p = p + lamport_xor_lane<1>(p);   // 求最大值时把 + 换成 fmaxf
```

原来取数用 `__shfl_xor`，在 gfx1250 上编译成 `ds_bpermute`：数据要送进 LDS 的交换电路转一圈再拿回来，每次几十到上百个时钟，5 轮串行。现在换成在寄存器之间直接搬数的指令：

- **xor 1/2/4/8**：`__builtin_amdgcn_update_dpp(i, i, 0x160 + N, 0xf, 0xf, false)`，生成 `v_mov_b32_dpp ... row_xmask:N`。
  - 32 个 lane 按 16 个一组分成两"行"，行内 lane i 读 lane (i xor N)。
  - `0x160 + N` 是硬件规定的 row_xmask:N 编码。
  - 两个 `0xf` 表示所有行、所有 lane 都启用。
- **xor 16**：跨两行，row_xmask 做不到，用 `__builtin_amdgcn_permlanex16(i, i, 0x76543210, 0xfedcba98, false, false)`。
  - 每个 lane 从另一行取一个值，两个选择码每 4 位对应一个位置。
  - `0x76543210` 表示位置 0~7 取对面行的第 0~7 个，`0xfedcba98` 表示位置 8~15 取第 8~15 个。
  - 合起来就是"位置不变、换到另一行"，等于 xor 16。

配对和相加顺序和 `__shfl_xor` 版完全相同，结果逐位一致。

### 4.2 跨 wave（第 4b~4d 步）

原来要同步两次：0 号线程一个人把 `partial[]` 加完，写进 LDS 的 `result`，再同步一次让大家来读。

现在同步一次后，每个线程自己按同样的顺序加：

```cpp
float r = partial[0];
for(int g = 1; g < row_groups; ++g)
    r += partial[g];     // 所有线程读同一地址，LDS 广播，没有冲突
```

求和与求最大值分别用两个模板实例，各有一份 `partial[32]`。这样快的 wave 写 amax 部分和时，不会覆盖慢 wave 还在读的平方和部分和。

**失败的写法**：用 `readlane` 在每个 wave 里串行加。所有 wave 都会挤在 CU 的标量单元（SALU）上排队，反而更慢。

### 4.3 效果和适用性

- **效果**：收益不大。2048×1 bf16 5.65 → 5.51，FP8 6.17 → 5.73。这是"DPP + 少一次同步"合在一起的效果，DPP 单独的收益没拆开测过。
- **能推广到别的 kernel 的条件**：
  - wave32 的架构（gfx10 以后）。gfx942 是 wave64，没有这些编码，不能照搬。
  - 32 个 lane 全部参与。DPP 从没在执行的 lane 取数，会拿到旧值。
  - warp reduce 处在关键路径上。如果 kernel 被访存吞吐卡住，看不出收益。

---

## 5. 优化三：一个线程负责多个包（PACKS），并禁止乘加合并

**代码**：kernel 模板参数 `PACKS`（第 1344 行起）；第 1373 行 `#pragma clang fp contract(off)`。

### 5.1 行为

原来一行只有两种排法：

- 一行一个 block，每线程 1 个包。hidden 8192 就是 1024 线程。
- 一行拆成多个 block 组成 cluster。

现在线程 t 负责 t、t+B、t+2B…… 共 PACKS 个包（B 是 block 的线程数）：

```cpp
tid[k] = tid0 + k * block;   // 线程 t 负责 t, t+block, t+2*block ...
```

每个包有自己的部分和。部分和在 LDS 里的位置是 `wave + k * (block/32)`，正好就是按列从左到右的顺序，所以第 4d 步的串行加顺序和原来一致。

**效果**：

- 4096 用每线程 2 个包（row2）最好。
- M=96~192 的台阶消失了，因为不再用 cluster，block 数等于行数，不会超过 CU 数。
- 大 hidden 下 row4 在 M=256 最好。

### 5.2 为什么要禁止乘加合并

改成多包之后，6144×300 第 68 行有 3 个元素和 car_fused 差一个最低位，`residual_out` 完全一致。查 ISA 发现：

- car_fused 算平方和是先 `v_pk_mul_f32` 平方，再单独 `v_add`，每步都舍入；
- 新代码形状变了以后，编译器把一部分"乘再加"合成了 `v_pk_fma_f32`，乘完不舍入直接加，结果就差一点。

合不合成由编译器看代码形状决定（默认 fp-contract=fast），所以必须显式禁止：

```cpp
#pragma clang fp contract(off)   // 放在 kernel 函数体最开头，否则编译报错
```

加上以后，PACKS = 2/3/4/8 全部逐位一致。

---

## 6. 优化四：发送和接收交给不同的 block（SPLIT，针对小 M、大 hidden）

**代码**：第 1400 行 `if constexpr(SPLIT)`（发送块）；第 1422 行（发送块顺手 rearm）；第 1868、2104 行（AR+add、AG 也写 rx 计数器）；第 3213 行（计数器分配加倍）。

### 6.1 为什么单 CU 慢

M=1、hidden 6144 时，一整行只有一个 block，在一个 CU 上。这个 CU 要先发出 2304 次跨卡写（768 个包 × 3 张卡），再反复读收件箱，读写走同一条访存通道。

消融数据（6144×1）：

| 变体 | us |
|---|---|
| 单 CU，跳过轮询和规约（只剩发送） | 6.28 |
| 单 CU，跳过规约 | 7.02 |
| 拆成 3 个 CU 的 cluster，跳过规约 | 6.31 |
| AR+add（12 个 64 线程的 block） | 5.57 |

说明把发送、接收分散到多个 CU 上，数据到得快。Gluon 本来就是发送程序和接收程序分开的。

时间戳实测（kernel 内部，相对第一个 wave 开始，单位 us）：

| 阶段 | 2048 row1 | 6144 row1 | 6144 cluster |
|---|---|---|---|
| 发送完成 | 0.56 | 0.80 | 0.88 |
| 轮询完成 | 3.20 | 5.04 | 3.60 |
| 规约完成 | 3.96 | 6.16 | 5.80 |
| 全部写完 | 4.48 | 6.64 | 6.24 |

（这是第 3 节改动之前测的。）

### 6.2 行为

grid 的前 `senders` 个 block 只负责发送，每个 block 用 `send_packs`（默认 64）个线程，其余线程直接退出。后面每行一个接收块，它的 PACKS 选"让 wave 数不超过 8"的最小值。

```
发送块：读轮次（tx 计数器） → 读输入 → 写给 3 张卡 → 把上一轮用过的收件箱填回 0xffff
       → tx 计数器加 1 → 退出
接收块：读轮次（rx 计数器） → 读本卡输入、残差、权重 → 轮询 → 求和 → RMSNorm → 写输出
       → rx 计数器加 1
```

### 6.3 两处协议改动

1. **轮次计数器拆成 tx、rx 两份。**
   - 发送块和接收块是不同的 block，没法保证"发送块读计数器"一定早于"接收块改计数器"。接收块等到数据就能结束，并不依赖本卡的发送块已经开始。
   - 拆开后各读各写，不会互相干扰。
   - 每次调用结束时 tx == rx。不拆分的 kernel（AR+add、AG、row 模式的 RMS）结束时两份都写，所以几个 kernel 交替调用不会乱。
   - 分配大小从 `slot/32` 个 uint32 变成 `2 * slot/32` 个（第 3213 行）。
2. **rearm 挪给发送块。** 第 r 轮要重新填的是 (r+2)%3 号收件箱，也就是上一轮用过的那块。上一轮的 kernel 已经结束，对方最多领先一轮，这一轮期间不会往这块写，所以发送块发完就可以填。这让接收块不用在结尾做 2304 次写，也把这些写分散到了多个 CU。

### 6.4 效果

下表是 bf16 的数。⚠ 可能受 GPU 1 被占用的影响，需要重测。

| hidden×M | row1 | split（send=32） | Gluon |
|---|---|---|---|
| 4096×1 | 7.13 | 6.46 | 6.55 |
| 6144×1 | 7.67 | 7.24 | 6.14 |
| 8192×1 | 8.26 | 7.47 | 7.79 |

- 2048 和 M ≥ 64 时 split 不如 row，dispatch 时要按形状选。
- 给 kernel 申请超过 160 KB 的动态 LDS（`AITER_LAMPORT_RMS_LDS`），可以强制一个 CU 只放一个 block。在加入"发送块做 rearm"之前测过，没有改善，说明问题不是 block 被挤到同一个 CU。

---

## 7. 优化五（小）：peer 指针改成标量参数；轮询只读另外 3 张卡

**代码**：第 1388 行 `all[8]`；第 1508 行（轮询）；第 1553 行（按卡号顺序求和）。

### 7.1 peer 指针改成标量参数

kernel 参数会预先放进寄存器（编译选项 `--amdgpu-kernarg-preload-count=32`），但遇到结构体就停了。原来 `LamportPeers peers` 是第 6 个参数，后面的 `slot`、`rank` 只能现场读。`peers.ptrs[(rank + i) % ngpus]` 这种按运行时下标取指针，每次发送前还要多等一次读取。

现在改成 `p0..p7` 八个单独参数，用固定下标 `all[g]` 取，和 AR+add kernel 的做法一样。发送前只剩轮次计数器这一次读取。收益约 0.1~0.4 us。

### 7.2 轮询只读 3 张卡

原来每轮轮询读 4 个格子，其中自己那格永远是 0xffff，读了再丢掉。现在只读 (rank+1)~(rank+3) 这 3 格，求和时再按卡号顺序拼回去：

```cpp
const int at = (g - rank - 1 + ngpus) % ngpus;   // ngpus-1 表示本卡
P v = local[k].data;
for(int i = 0; i < ngpus - 1; ++i)
    v = at == i ? scratch[k][i].data : v;
```

结果仍然逐位一致。⚠ 效果没测准，那次正赶上 GPU 1 被占用，row1 反而显得变慢，需要在干净的机器上重测，再决定留不留。

---

## 8. 试过、没用的

| 尝试 | 结果 | 原因 |
|---|---|---|
| cluster 的部分和用带返回值的原子交换（`atomic_exchange`）发布，避开 storecnt | 9~35 us，大幅变慢 | 原子操作排在前面的跨卡写后面，还有同一 cache line 的竞争 |
| cluster 里一半 wave 只发送、发完先退出，另一半接收 | 比原版 cluster 更慢 | cluster 规约本身要走一趟显存（写确认 + `global_inv` + 读），约 1.5~3.8 us，省不下来 |
| 跨 wave 串行和用 `readlane` | 变慢 | 所有 wave 挤在 SALU 上 |
| 第一版 SPLIT（接收块做 rearm、规约还是旧写法） | 比 row1 慢 | 接收块尾部 2304 次 rearm 写，加上旧规约 |
| 在 kernel 最开头判断调试参数 `dbg` | 每次调用多约 2.5 us | `dbg` 在参数列表末尾，没有预载入寄存器，kernel 一开始就要等它读回来，挡在发送前面 |

---

## 9. 6144 小 M 还差的 1 us：线索

- Gluon 接收端每个 lane 每轮读 12 个 16 字节行，和我们 split 接收块（8 个 wave × 3 个包 × 3 张卡）一样多。但 Gluon 用 3 个值加 1 个标签打包，**不需要 rearm**，接收程序只写输出。
- 我们现在接收块还要读本卡输入（`local`）用于求和。可以考虑让发送块把本卡数据也写一份到本卡收件箱，或者干脆用本卡输入的原始地址，省掉一次依赖。
- 可以考虑在 FP8 时把求平方和与求最大值合并成一次同步，但最大值依赖 inv_rms，要想办法。
- 时间戳方法：把 `LAMPORT_STAMPS` 宏打开，用 `scripts/stamps.py`。**不要**用运行时参数控制插桩，原因见第 8 节最后一行。

---

## 10. 工具与注意事项

`pr8022_wip/scripts/` 里的文件，原件在 `/root/.claude/projects/-app-aiter/notes/pr8022/`：

| 文件 | 用途 |
|---|---|
| `bench.py` | graph 计时。`-c lamport gluon add` 选比较对象（add 是 AR+add，当下限参考）；`-q 0 1` 选 bf16/FP8；`--rounds` 跑几轮；自动开 `AITER_GLUON_FUSED_AR=1` |
| `summ.py` | 合并多份 bench 日志成对比表：`summ.py 基线日志 名字=日志 ...` |
| `variants*.sh` | 用环境变量批量扫变体（v3 版的格式是 `名字:CLUSTER:PACKS:CBS`） |
| `dbg_diff.py` | 定位和 car_fused 的逐位差异：`dbg_diff.py M hidden 次数` |
| `stamps.py` | 读 kernel 内时间戳（需打开 `LAMPORT_STAMPS`） |
| `build.py` | 触发 JIT 编译 |

临时环境变量（第 2194 行，正式版要删）：

| 变量 | 含义 |
|---|---|
| `AITER_LAMPORT_RMS_CLUSTER` | 0：强制 row；1：强制 cluster |
| `AITER_LAMPORT_RMS_PACKS` | 每线程包数 |
| `AITER_LAMPORT_RMS_CBS` | cluster 每块包数 |
| `AITER_LAMPORT_RMS_SPLIT` | 1：用 SPLIT 模式 |
| `AITER_LAMPORT_RMS_SENDPACKS` | SPLIT 发送块每块包数 |
| `AITER_LAMPORT_RMS_LDS` | 动态 LDS 字节数 |

**踩过的坑：**

- 改了 .cuh 不删 `.so`，JIT 不会重编，会测到旧代码。
- gfx1250 没有 `__builtin_amdgcn_s_waitcnt`，要用 `asm volatile("s_wait_storecnt 0x0")`。
- rocprofv3 包着多进程测试会卡住，超时后留下的子进程会一直往被删除的日志里写，把磁盘写满。别用它跑多卡测试；真要用的话，结束后检查 `ps` 有没有残留。
- 跑之前先 `amd-smi process` 看有没有别的容器在用 GPU。
