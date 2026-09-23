# AllGather + GEMM 通算融合算子 学习笔记

> 环境:8× AMD MI300X(架构代号 gfx942 / CDNA3),单节点。
> 代码:`amd-distributed/all-gather-gemm/`,比赛第 1 名实现。
> 说明:本文所有硬件术语在首次出现时用括号/破折号解释,不假设读者有底层背景。

---

## 目录

1. [代码结构总览](#1-代码结构总览)
2. [nvim 环境配置(高亮 + 补全)](#2-nvim-环境配置高亮--补全)
3. [config 文件的意义 + 支持哪些 shape](#3-config-文件的意义--支持哪些-shape)
4. [XCD remap 是什么、作用在哪](#4-xcd-remap-是什么作用在哪)
5. [核心:跨卡通信的完整硬件通路](#5-核心跨卡通信的完整硬件通路)
6. [cache 基础 + sc0/sc1 开关 + Triton modifier 对照表](#6-cache-基础--sc0sc1-开关--triton-modifier-对照表)
7. [为什么是 fine-grained 而不是整块 uncached](#7-为什么是-fine-grained-而不是整块-uncached)
8. [bypass cache 真的慢很多吗?(latency vs bandwidth)](#8-bypass-cache-真的慢很多吗latency-vs-bandwidth)

---

## 1. 代码结构总览

### 任务语义
每张卡持有 `[local_M, K]` 输入和 `[local_N, K]` 权重。先把 8 张卡的输入 allgather 成 `[M=local_M*8, K]`,再 `full_input @ weight.T`,输出 `[M, local_N]`。

### 文件角色

| 文件 | 作用 |
|---|---|
| `task.py` / `reference.py` | 比赛给定的任务定义、参考实现、正确性校验(别改) |
| **`ref10_first.py`**(1646 行) | **主实现**,入口 `custom_kernel`,融合 kernel `triton_mm_kernel` |
| `ref10_first.hip` | HIP/C++ 端:IPC 共享内存建立、跨卡指针交换 |
| `ref10_first.cpp` | PyTorch 自定义算子注册(`my_ops.init/put_kernel/clear`) |
| `gen.py` | 把 `.hip`/`.cpp` 源码**内联**进 `ref10_first.py`,生成单文件提交 |
| `run.py` | 本地 8 卡 spawn 跑各种 shape 做正确性校验 + profiling |

### 设计骨架:单 kernel 融合,CTA 分工
一个 persistent(常驻)Triton kernel,把 grid 里的 block(CTA)分成两类,用 `COMM_PIDs = 7 * SEND_CTA_NUM` 划界:

- **通信 CTA**(前 `7*SEND_CTA_NUM` 个):把本卡输入 push(推)到其他 7 张卡的 IPC 显存,发完置信号量(flag)。
- **计算 CTA**(其余):
  - 若要算的数据是本卡的 → 直接读本卡输入;
  - 若是远端的 → **自旋等对应 flag** → 数据到位后做 `tl.dot` → 加 bias → 写出 C。

这样通信和计算在 GPU 上真正 overlap(重叠),而不是「先 allgather 完再算」。

**关键模型:push(写)+ 本地读。只有写穿越卡间链路,读永远是本地的。**

---

## 2. nvim 环境配置(高亮 + 补全)

### 问题诊断
- **hip 没高亮**:`.hip` 后缀没有 filetype 映射 → 不挂语法 parser。(`.cu/.cuh` 之前已映射成 `cuda`,其实有高亮。)
- **补全变量消失**:整机**没装 clangd**(C/C++ 的语言服务器,提供语义补全)。「前面写过的变量」这种词补全来自 blink.cmp 的 buffer 源,和文件格式无关,一直都在;但成员/类型/跨符号的**语义补全需要 clangd**。

### 做的两件事
1. **filetype 映射**:改 `~/.config/nvim/lua/config/autocmds.lua`,把 `cu`/`cuh`/`hip` 三种后缀都识别成 `cpp`(cpp 的 treesitter parser 已装,高亮直接可用;这些文件里没有未注释的 `<<<>>>` 启动语法,cpp parser 解析无碍)。
2. **装 clangd**:在 `~/.config/nvim/lazyvim.json` 加 `lazyvim.plugins.extras.lang.clangd` extra,再用 mason 装好 clangd 二进制(位于 `~/.local/share/nvim/mason/bin/clangd`)。

> 补充:要让 `#include <torch/...>`、HIP 头文件也能解析(跨文件跳转),clangd 需要一个 `compile_commands.json`(编译数据库,告诉它 include 路径和 flag)。没有的话头文件飘红,但不影响本文件符号补全。

---

## 3. config 文件的意义 + 支持哪些 shape

### 为什么 GEMM 需要 config
GEMM 把大矩阵切成小块(tile)喂给 MI300X 的 MFMA(矩阵乘加硬件单元)。**没有一套分块参数能对所有 shape 最优**:块太大→占寄存器多、并行度低;块太小→算力利用率低。做法是**离线 autotune 每个 shape 找最优参数,硬编码进 `online_config`**;跑到某 shape 查表,命中就用融合 kernel,没命中 fallback 到纯 torch 或 split-k。

### config 字段含义(以 `(4096, 512, 4096)` 为例)
```python
{'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256,   # 每个 CTA 算 C 的 64×64 块,K 方向一次读 256
 'num_warps': 8, 'num_stages': 2,                 # 8 个 warp;软件流水 2 级(预取重叠)
 'waves_per_eu': 2,                               # 每个执行单元驻留 2 个 wave,控制占用率
 'matrix_instr_nonkdim': 16, 'kpack': 1,          # MFMA 指令形状(16×16)、K 打包方式
 'SLEEP_CYCLE': 2, 'SPLIT_BLOCK': 4}              # ← 融合专属:自旋等信号睡多久 / 发送切几片
```
前面是**标准 GEMM 调优参数**;后两个是**这个融合算子特有的通信参数**。

### 支持哪些 input shape
`online_config` 的 key 是 `(M, N, K)`,其中 `M = local_M×8`(allgather 后全局 M)、`N = n/8`(本卡权重行数)、`K = k`。反解成原始 `(m, n, k)`:

| key (M, N, K) | 原始 (m, n, k) | 说明 |
|---|---|---|
| (64, 2304, 7168) | (64, 18432, 7168) | 空 `{}`,M=64 太小,跳过融合 |
| (512, 1536, 4096) | (512, 12288, 4096) | 融合 |
| (2048, 360, 2880) | (2048, 2880, 2880) | 融合 |
| (4096, 512, 4096) | (4096, 4096, 4096) | 融合 |
| (8192, 1792, 4096) | (8192, 14336, 4096) | 融合 |
| (8192, 3696, 8192) | (8192, 29568, 8192) | 融合 |

**只有命中这张表的 shape 走最快融合路径**,其余 fallback。另有特判:`m=8, k=7168` 走 split-k;`m=64` 跳过融合。
(注:`run.py` 里的 config 是本地正确性测试 shape,和 `online_config` 不是同一套。)

---

## 4. XCD remap 是什么、作用在哪

MI300X 一颗 GPU 有 **8 个 XCD(计算 die,每块有自己的 L2 缓存)**。默认 pid→tile 的线性映射会让相邻 tile 分散到不同 XCD、读不到彼此 L2 里缓存的数据。**XCD remap 把「相邻 pid」重排到同一个 XCD 上,提升 L2 命中率和复用。**

- **作用范围**:所有**计算 CTA**(本卡数据 + 远端数据两段都做);通信 CTA 不做。
- **它优化的是「读侧本地 L2 命中」**,和「数据当初怎么跨卡过来」是两个互不干扰的阶段。
- 和 buffer 是否 uncached 无关——那是通信可见性问题,这是计算局部性问题。

---

## 5. 核心:跨卡通信的完整硬件通路

**关键认知:push(写)+ 本地读模型。只有写穿越 fabric,读永远本地。**

### 拓扑与地址映射
- 8 张卡**全互联**:每张有 7 条 XGMI(Infinity Fabric,卡间高速直连链路)直连另外 7 张。
- `hipIpcOpenMemHandle` 把对端 GPU 的 HBM(显存)映射进本卡的虚拟地址空间;对映射地址做 store,硬件自动路由到 XGMI。

### 片上缓存层级(从近到远)
```
CU(计算单元,VGPR) → vL1(每 CU 小抄本) → L2(每 XCD 中抄本,几 MB)
                    → Infinity Cache(整卡大抄本) → HBM(显存原件)
                                                  └ 经 XGMI 可达对端
```

### 发送侧(通信 CTA 把本卡输入推给远端)
```python
val = tl.load(src, cache_modifier=".cv")        # (a) 从本卡 HBM 读源数据进 VGPR
tl.store(remote_addr, val, cache_modifier=".wt")# (b) 写穿透 → XGMI → 对端 HBM
tl.atomic_add(remote_flag, 1, sem="release", scope="sys")  # (c) 置信号(release,保证数据先可见)
```

### 接收侧(计算 CTA 消费远端数据)—— 全程读本地
```python
A_ptr += my_rank_base                             # 指向"我自己"的 buffer
result = tl.load(flag_ptr, cache_modifier=".cv")  # (a) 轮询本地 flag(别卡写进了我的 HBM)
while result != SEND_CTA_NUM: ...
# (b) 数据已落在我的本地 HBM:
a = tl.load(A)                                    # ← 普通 cached 读,喂 MFMA
acc = tl.dot(a, b, acc)
```

### 为什么 push + 本地读
- **写方**:连续成块流式写对端(XGMI 喜欢大块顺序流量)。
- **读方**:计算时以**本地 HBM 延迟**取数喂 MFMA;若改成计算时跨卡 pull,每个 tile 取数吃一次 XGMI 往返,MFMA 频繁 stall。

**跨 fabric 的只有两样:数据 store 和 flag 原子写。其余全在各自卡内。**

---

## 6. cache 基础 + sc0/sc1 开关 + Triton modifier 对照表

### 两个麻烦(所有 modifier 要解决的)
1. **读到旧的(stale)**:别人改了 HBM 原件,你手边抄本还是旧值。
2. **写卡在本地**:你只改了自己抄本,还没落回原件,别人看不到。

### 硬件开关 sc0 / sc1
CDNA3 里**每条读/写指令自带两个开关位 `sc0`/`sc1`**(老 GPU 叫 `glc`/`slc`),合起来是一个「作用范围(scope)」旋钮,2 位 4 个档:

| 档位 | 保证「谁」看到最新值 | 代价 |
|---|---|---|
| wavefront(一波 64 线程) | 只自己这波 | 最快 |
| workgroup(一个线程块) | 同一线程块 | 快 |
| agent(整张卡) | 整卡一致 | 中 |
| **system(全节点跨卡)** | **8 卡都看到最新** | 最慢,跨卡必须用 |

**旋钮越大 = 越往 HBM/跨卡走 = 越慢但越多人看到最新。**

### Triton modifier 对照表

**读(load):**
| 写法 | 直译 | 大白话 | 档位 | 代码里 |
|---|---|---|---|---|
| `.ca` | cache all | 默认,存 L1+L2,假设没人改它 | 最低 | 读权重 B |
| `.cg` | cache global | 跳过 L1,只在 L2 存 | agent | 写 C 时用 |
| **`.cv`** | cache volatile(易变) | **不信任何抄本,直接取最新** | **system** | **轮询 flag、读别卡数据** |

**写(store):**
| 写法 | 直译 | 大白话 | 档位 | 代码里 |
|---|---|---|---|---|
| `.wb` | write back | 默认,先写抄本当脏数据待着,以后才落 HBM | 最低 | **故意不用** |
| `.cg` | cache global | 写到 L2 层 | agent | 写输出 C |
| **`.wt`** | write through(写穿) | **写的当下穿透所有抄本送到对端/内存,本地不留脏** | **system** | **推数据给别卡** |

---

## 7. 为什么是 fine-grained 而不是整块 uncached

### 两个正交维度(消除误解的钥匙)
- **fine-grained vs coarse-grained**:**一致性**属性——写能不能在 kernel 运行中跨卡可见。
- **cached vs uncached**:**可缓存性**属性——访问走不走 cache。

fine-grained **不等于**「跨卡读被缓存成 stale」;它只是**赋予内存「能做到细粒度可见」的能力**,每次访问走 cache 还是穿透,由指令 modifier 精确挑。

### GPU 没有硬件跨卡一致性,也没有全量 flush
CPU 的 MESI 是硬件自动 snoop;**GPU 的 L2 之间不互相 snoop**,跨卡可见性全靠指令上的 cache 控制位在指定 scope 下显式驱动。这些是**逐指令、逐地址的定向操作**,成本正比于实际读写的数据,**不是扫全 L2 的昂贵 flush**。

### 命门:收到的数据 = GEMM 的输入,是同一块内存
在这个融合 kernel 里,rank i 推给 rank j 的数据,**落地后立刻就是 rank j 做 GEMM 的 A 输入,同一块内存,无中转**。而读 A 用的是**普通 cached load**(`a = tl.load(A)`,无 modifier)。

- 内存属性(cached/uncached)是**一刀切**的,盖过指令 modifier。
- 若把这块设成 **uncached**,GEMM 那句读 A 也被迫 uncached → matmul 取数全绕 cache → **算力崩**。
- 要用 uncached scratch,就得多一次「scratch → 计算 buffer」的拷贝(几百 MB 过一遍 HBM),**这正是融合要消灭的开销**。

### 结论
- 「权重 + 本卡数据」用普通 hipMalloc(缓存)完全正确——**代码就是这么做的**,fine-grained 那块只装「收到的跨卡数据 + flag」。
- 但「通信 buffer」在融合里**同时是通信落地区和 GEMM 输入区**,一块内存要被两种截然不同的访问方式对待。
- **只有 fine-grained + 逐指令 modifier 能「写它时穿透、算它时走 cache」;uncached 一刀切做不到。**

---

## 8. bypass cache 真的慢很多吗?(latency vs bandwidth)

**会,而且是数量级的差别——但原因不是「每次 load 的延迟」,而是「cache 复用没了,HBM 流量翻几倍」。**

### 两种「读」别混为一谈
| | 读 flag | 读数据(算矩阵) |
|---|---|---|
| 本质 | **latency**:等一个会变的值 | **bandwidth**:搬海量字节喂 MFMA |
| 量 | 几个 int | 每 tile **几十 MB**,K 越大越多 |
| 该 bypass 吗 | **该**(`.cv`) | **不该** |

### cached 数据读**不需要每次 flush**
数据一旦落地、flag 读到,这一轮**它不再被改写(稳定)**。GEMM 第一次读进 L2,之后**反复复用,零 flush**。要 bypass/flush 的只有「会变的东西」:跳变的 flag、跨卡的写。

### 慢多少:复用没了,流量爆炸
矩阵乘法里,同一条 A 行带 `A[BLOCK_M, K]` 会被**同一行所有 N/BLOCK_N 个 tile 反复读**。

真实 shape 举例(N=29568, BLOCK_N=256):
- 一条 A 行带被 **≈115 个 tile 共用**。
- **走 cache**:留在 L2,115 个 tile 大多命中,从 HBM 只搬一遍左右。
- **uncached**:每 tile 从 HBM 重搬 → 同一份 A 被拉近 **115 次** → A 的 HBM 流量翻两个数量级。

每 tile 光 A 就读 `BLOCK_M × K`(K = 4096~30720,单 tile 几十 MB)。丢掉复用 = 凭空给 HBM 加几倍到几十倍搬运量 → **把 compute-bound(算力瓶颈)的 GEMM 拖成 memory-bound(带宽瓶颈)**。
> `XCD remap` 整套设计的意义**就是榨这个 L2 复用**;buffer 一旦 uncached,这套全白做。

### 关于「flag 轮询是 latency 大头」
你的直觉**在「flag 等待不可掩盖」这个前提下**才成立。但这个融合 kernel 的**全部功夫**就是消灭这个前提:
- 计算 CTA 先算**本卡本地 tile**(不等任何 flag),把通信时间**藏在本地计算背后**(overlap);
- 通信也切片、边到边算(SPLIT_BLOCK)。

一旦 flag 等待被 overlap 藏住,**瓶颈就回到 GEMM 稳态吞吐**——也就是数据读的带宽。这时 cache 复用就是决定性的。**正因为设计成功把 flag 等待压下去,数据读的 cache 效率才成了主角。**

---

## 一句话总纲

> AG-GEMM 的赢点:用 IPC 打通 8 卡显存 → 一个常驻 Triton kernel 内 CTA 分工(通信 CTA 写穿推数 + 计算 CTA 自旋等信号后开算)→ **push+本地读** + **本地 tile 优先 + XCD remap** 让通信/计算充分 overlap。
> 内存策略的精髓:通信落地区必须是 **fine-grained(可缓存 + 跨卡可见能力)**,靠**逐指令 cache modifier**做到「跨卡写它时穿透、GEMM 算它时走 cache」——这是整块 uncached 一刀切给不了的。
