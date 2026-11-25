# Lamport AR + RMSNorm kernel 讲解文档（含源码）

> - 文件：`/app/aiter/csrc/include/custom_all_reduce_gfx1250.cuh`，分支 `gfx1250_lamport_rms_opt`，提交 `3b15a6366 [HIP] fea: gfx1250 Lamport AR + RMSNorm: split sends, preloaded arguments, faster reductions`（PR #8559）
> - 每一节先列出源码（左边是文件里的行号），后面是讲解
> - 源码由脚本 `notes/rms_opt_1123/scripts/gen_walkthrough.py` 从文件里直接截取，和仓库一致
> - 更早的版本（对应 abb37852a，含 cluster 和实验开关）备份在同目录 `kernel_walkthrough_1124_abb37852a.md`

---

## 0. 先看全局

### 0.1 这个 kernel 做什么

每张卡有一份 `input[rows][hidden]`（bf16 或 fp16），一次 kernel 完成：

```
ar       = 4 张卡的 input 按卡号顺序相加（fp32 累加），舍入成 bf16
res_out  = ar + residual_inp（fp32 加）          → 写 residual_out
out      = res_out * rsqrt(mean(res_out^2) + eps) * weight
           bf16 输出：直接写 output
           FP8 输出：按行求 amax，量化成 FP8，写 output 和 scale_out
```

结果和 `ar_rmsnorm_gfx1250`（car_fused）逐位一致，每一步的运算顺序都照抄它。
"先把 all-reduce 结果舍入成 bf16 再加 residual"也是照抄：不融合的路径里 all-reduce 输出本来就是 bf16 张量，这样不管线上走哪条路径，结果都一样。

### 0.2 几个基本单位

| 名词 | 含义 |
|---|---|
| 包（pack） | 16 字节 = 8 个 bf16，一次读写就是一个包 |
| row_packs | 一行有多少个包 = hidden / 8，例如 6144 → 768 |
| wave | 32 个线程一起执行 |
| 一组包（group） | 连续 32 个包，每组有自己的轮次计数器 |
| block | 一个 CU 上运行的一组 wave |

### 0.3 收件箱

```
每张卡：inbox[轮次缓冲 buf 0..2][发送方 sender 0..3][包 0..slot-1]，每格 16 字节；初始化时整块填 0xff
inbox[buf][sender] 这一段叫一"格"（slot），一格有 slot 个包
```

- 第 r 轮，卡 A 把第 p 个包写到每张别的卡 B 的 `inbox[r%3][A][p]`；卡 B 轮询自己这一格，还有 0xffff 就是没到。
- **3 块缓冲轮换**：最快的卡最多领先一轮。第 r 轮用缓冲 r%3，同时把缓冲 (r+2)%3（上一轮用过的）重新填成 0xffff，叫**回填**。
- **轮次计数器**：每组包一个。所有卡调用序列相同，所以同一组包的计数器在每张卡上都一样，不需要卡间同步。
- **tx / rx 两份计数器**：split 模式里发送和接收是不同的 block，各用一份。其他所有 Lamport kernel 每次两份都写，保证两份始终相等。

### 0.4 两种模式

```
row 模式：每行一个 block，同一个 block 既发送又接收
  读本卡数据 → 发给 3 张卡 → 等一次参数读 → 读 residual/weight → 轮询 → 计算 → 回填、更新轮次

split 模式：发送和接收分开
  grid = [发送块 0 .. senders-1][每行一个接收块]
  发送块：读 tx 轮次 → 读本卡数据 → 发给 3 张卡 → 回填 → tx+1 → 退出
  接收块：读 rx 轮次 → 读本卡数据、residual、weight → 轮询 → 计算 → rx+1
```

小 M 时一个 CU 既要发一整行又要轮询，是瓶颈，所以把发送分散到多个发送块（split）；M 大时 row 模式更好。由 host 端自动选择（第 9 节）。

---

## 1. 顶部说明和常量
```cpp
1358  // ---------------------------------------------------------------------------
1359  // Lamport one-shot fused allreduce + residual add + RMSNorm (2-byte dtypes)
1360  // ---------------------------------------------------------------------------
1361  /*
1362   * No barrier and no copy-in: every 16B pack of the input is pushed into
1363   * slot[rank] of each peer's inbox, and its receiver polls its own inbox until
1364   * no half-word is the 0xffff sentinel (a bf16/fp16 NaN; the sender rewrites
1365   * its own 0xffff to 0x7fff). The epilogue then follows ar_rmsnorm_gfx1250 step
1366   * by step (rank-order fp32 sum, round to T, add the residual, 32-lane butterfly
1367   * then a serial sum over the wave groups of the row), so out and residual_out
1368   * match it bit for bit, and so do the FP8 out and scale of its per-token quant.
1369   *
1370   * Inbox (one IPC buffer per rank, memset to 0xff at init):
1371   *   inbox[kLamportBuffers][ngpus][slot] of 16B, slot = the most packs a call takes
1372   * Each group of 32 packs has its own round counters (rank-local), which pick
1373   * the buffer. Every rank runs the same calls, so the counter of a pack is the
1374   * same on every rank whatever the row width and block shape of each call, and
1375   * a pack that sits out a call keeps its counter and its inbox. A peer is at
1376   * most one round ahead on a pack, so buffer (c+2)%3 is re-armed in round c and
1377   * the kernel boundary orders that before my next round's sends.
1378   *
1379   * One block a row receives the row and reduces it on chip; thread t owns packs
1380   * t + k * blockDim.x (k < PACKS) of its row. The packs are sent
1381   *   - row mode: by that same block, before it polls;
1382   *   - split mode: by sender blocks 0 .. senders - 1, ahead of the row blocks in
1383   *     the grid. They spread the remote stores over more CUs and also re-arm.
1384   *     Senders and receivers count rounds apart (tx and rx counters).
1385   * LamportArRms::run picks the mode and PACKS.
1386   */
1387  constexpr int kLamportBuffers       = 3;
1388  constexpr int kLamportWave          = 32;   // = gfx1250 warpSize
1389  constexpr int kLamportRowMaxThreads = 1024; // a row is one block: hidden <= 8192
```

- 1361–1385 是 kernel 的总说明（第 0 节就是它的中文版）。
- `kLamportBuffers = 3`：轮换的缓冲数。`kLamportWave = 32`：一个 wave 的线程数。`kLamportRowMaxThreads = 1024`：一行最多 1024 个包，所以 hidden ≤ 8192。

## 2. 收件箱大小、peer 指针
```cpp
1391  // slot: the most 16B packs a call takes, a multiple of kLamportWave
1392  inline constexpr int64_t lamport_ar_rms_buffer_bytes(int ngpus, int64_t slot)
1393  {
1394      return (int64_t)kLamportBuffers * ngpus * slot * 16;
1395  }
1396  
1397  // The exact all-gather's escape inbox, one uint32 a pack, follows the data
1398  // inbox (see lamport_ag_gfx1250); armed with 0, the data inbox with 0xff.
1399  inline constexpr int64_t lamport_escape_bytes(int ngpus, int64_t slot)
1400  {
1401      return (int64_t)kLamportBuffers * ngpus * slot * 4;
1402  }
1403  
1404  // The two-shot all-gather inbox, [ngpus][slot] of 16B, one buffer (no rounds),
1405  // follows the escape inbox; armed with 0xff. A two-shot call takes up to
1406  // ngpus x slot packs, so a rank's chunk, ceil(packs / ngpus), is up to slot
1407  // packs (see lamport_ar_2shot_gfx1250).
1408  inline constexpr int64_t lamport_two_shot_bytes(int ngpus, int64_t slot)
1409  {
1410      return (int64_t)ngpus * slot * 16;
1411  }
1412  
1413  // Byte offset of the two-shot all-gather inbox in a rank's Lamport buffer.
1414  inline constexpr int64_t lamport_two_shot_offset(int ngpus, int64_t slot)
1415  {
1416      return lamport_ar_rms_buffer_bytes(ngpus, slot) + lamport_escape_bytes(ngpus, slot);
1417  }
1418  
1419  // A rank's whole Lamport buffer: the data inbox, the escape inbox, then the
1420  // two-shot all-gather inbox.
1421  inline constexpr int64_t lamport_buffer_bytes(int ngpus, int64_t slot)
1422  {
1423      return lamport_two_shot_offset(ngpus, slot) + lamport_two_shot_bytes(ngpus, slot);
1424  }
1425  
1426  struct __align__(16) LamportPeers
1427  {
1428      void* ptrs[kMaxNgpusAr];
1429  };
```

- `lamport_ar_rms_buffer_bytes`：数据收件箱的大小 = 3 × 卡数 × slot × 16 字节，本 kernel 只用这一块。
- 后面的 escape 区和 two-shot 区是 all-gather、2-shot all-reduce 用的，接在数据收件箱后面，同一次分配。
- `LamportPeers`：host 端保存每张卡收件箱地址的结构体。kernel 不再直接收这个结构体（见第 7 节）。

## 3. 哨兵 0xffff
```cpp
1431  // one bit per half-word that is the 0xffff sentinel
1432  DINLINE uint32_t lamportSentinelMask(uint32_t w)
1433  {
1434      uint32_t x   = ~w;
1435      uint32_t tmp = ((x & 0x7fff7fff) + 0x7fff7fff) | x;
1436      return (tmp & 0x80008000) ^ 0x80008000;
1437  }
1438  
1439  // 0xffff -> 0x7fff, both NaN, so sent data is never the sentinel
1440  DINLINE uint32_t lamportCanonicalize(uint32_t w) { return w ^ lamportSentinelMask(w); }
```

- `lamportSentinelMask(w)`：一个 uint32 里有两个 16 位，哪个正好是 0xffff，就把那个 16 位的最高位置 1，其他位都是 0。结果不为 0 就说明还有数据没到。
- `lamportCanonicalize(w)`：把数据里恰好是 0xffff 的 16 位改成 0x7fff（异或掩码，正好翻转最高位）。0xffff 和 0x7fff 在 bf16 里都是 NaN，改了不影响数值意义，但真实数据永远不会被误认成"没到"。发送前每个包都要过一遍。

## 4. 读写函数
```cpp
1442  using lamport_u32x4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
1443  
1444  DINLINE void lamport_store_b128(void* dst, lamport_u32x4 v)
1445  {
1446  #if AITER_GFX1250_HAVE_B128
1447      __builtin_amdgcn_global_store_b128((llx_v4u_gptr)dst, v, "");
1448  #else
1449      __builtin_nontemporal_store(v, reinterpret_cast<lamport_u32x4*>(dst));
1450  #endif
1451  }
1452  
1453  DINLINE lamport_u32x4 lamport_load_b128(const void* src)
1454  {
1455  #if AITER_GFX1250_HAVE_B128
1456      return __builtin_amdgcn_global_load_b128((llx_v4u_gptr)src, "");
1457  #else
1458      return __builtin_nontemporal_load(reinterpret_cast<const lamport_u32x4*>(src));
1459  #endif
1460  }
1461  
1462  // Slot (buf, sender) of an inbox, inbox[buf][sender][slot] of 16B packs. The
1463  // caller passes uniform values, so the result is in SGPRs.
1464  DINLINE char* lamport_slot(void* inbox, uint32_t buf, int sender, int ngpus, int slot)
1465  {
1466      return static_cast<char*>(inbox) + (int64_t)(buf * ngpus + sender) * slot * 16;
1467  }
1468  
1469  // Pack `pack` of a slot, as a system-scope 16-byte store or load that bypasses
1470  // the caches (th:TH_*_BYPASS scope:SCOPE_SYS, cache policy 27), as the Gluon
1471  // all-reduce: a buffer instruction, the global ones take no temporal hint. The
1472  // byte offset is from the slot, so it fits an int (a slot is under 2 GiB, see
1473  // init_lamport_ar_rms; the whole inbox may not be).
1474  DINLINE void lamport_store_sys(char* slot_base, int pack, lamport_u32x4 v)
1475  {
1476  #if AITER_GFX1250_HAVE_B128
1477      __builtin_amdgcn_raw_buffer_store_b128(
1478          v, __builtin_amdgcn_make_buffer_rsrc(slot_base, 0, 0x7fffffff, 0), pack * 16, 0, 27);
1479  #else
1480      lamport_store_b128(reinterpret_cast<lamport_u32x4*>(slot_base) + pack, v);
1481  #endif
1482  }
1483  
1484  DINLINE lamport_u32x4 lamport_load_sys(char* slot_base, int pack)
1485  {
1486  #if AITER_GFX1250_HAVE_B128
1487      return __builtin_amdgcn_raw_buffer_load_b128(
1488          __builtin_amdgcn_make_buffer_rsrc(slot_base, 0, 0x7fffffff, 0), pack * 16, 0, 27);
1489  #else
1490      return lamport_load_b128(reinterpret_cast<lamport_u32x4*>(slot_base) + pack);
1491  #endif
```

- 1444–1460 `lamport_store_b128` / `lamport_load_b128`：普通的 16 字节读写（scope:SCOPE_SYS），1-shot、2-shot 等别的 kernel 在用。
- 1462–1467 `lamport_slot`：算出"格" `inbox[buf][sender]` 的起始地址。调用时传进来的值整个 wave 都相同，所以结果在 SGPR 里。
- 1469–1491 `lamport_store_sys` / `lamport_load_sys`：本 kernel 的跨卡写、轮询读、回填都用它们。
  - 用 buffer 指令：`make_buffer_rsrc` 以"格"起点生成描述符，偏移 = `pack * 16`，最后一个参数 `27` 是 cache 策略。
  - `27` = TH 3（BYPASS）+ scope 3（SYS）× 8。生成的指令是 `th:TH_*_BYPASS scope:SCOPE_SYS`，和 Gluon 一样：SYS 保证跨卡可见，BYPASS 提示 cache 不要保留这些只用一次的数据。global 指令的 builtin 只能设 scope，设不了 TH，所以要用 buffer 指令。
  - 偏移是 32 位的。从"格"起点算，最多一格的大小；从整个收件箱起点算的话，容量调得很大时会超过 2 GiB。
  - `#else` 是其他架构的写法：这些 kernel 只在 gfx1250 上运行，其他架构只要求能编译通过。

## 5. 同步和等待
```cpp
1493  
1494  // A block barrier that orders LDS only: __syncthreads() also waits for every
1495  // outstanding global store of the wave, here the remote sends, whose acks take
1496  // about one more fabric trip than the peers' data took to arrive.
1497  DINLINE void lamport_lds_barrier()
1498  {
1499      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
1500      __builtin_amdgcn_s_barrier();
1501      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
1502  }
1503  
1504  // Waits for a kernel argument past the preloaded ones (offset 128) to come
1505  // from memory. A block that sends a row and then polls the inbox gets the
1506  // peers' data sooner when its first poll waits for this load: TP4 bf16 row of
1507  // 6144, one block a row, 10.4 -> 7.6 us. An s_sleep of any length does not
1508  // give the same gain (the load seems to return after the sends left).
1509  DINLINE void lamport_kernarg_wait()
1510  {
1511      const __attribute__((address_space(4))) volatile int* args =
1512          (const __attribute__((address_space(4))) volatile int*)
1513              __builtin_amdgcn_kernarg_segment_ptr();
1514      const int v = args[128 / 4];
1515      asm volatile("" ::"s"(v));
1516  }
```

- `lamport_lds_barrier`：block 内同步，只保证 LDS 可见（fence 的第二个参数 `"local"` 表示只管 LDS）。`__syncthreads()` 还会等这个 wave 发出的所有显存写收到确认，包括刚发出的跨卡写，要多等一个来回（约 1 us）。这里同步只是为了交换 LDS 里的部分和，用轻量版就够了。
- `lamport_kernarg_wait`：读参数区偏移 128 处的参数并强制等它返回（这个位置不在预载范围内，要真的去显存读）。只在 row 模式发送之后调用一次。实测让 row 模式快很多（6144×1：10.4 → 7.6 us），固定时长的 sleep 替代不了。机制没有证实，推测是这次读要排在刚发出的跨卡写后面才能返回，相当于"等写发出去了再开始轮询"。

## 6. 规约
```cpp
1518  // i of lane ^ OFF (wave32) in registers: DPP row_xmask within a row of 16,
1519  // permlanex16 across the two rows. __shfl_xor goes through ds_bpermute, an
1520  // LDS round trip per step. A row_xmask lane always reads a lane of its row, so
1521  // bound_ctrl changes no value; it lets the compiler fold the move into the
1522  // add or max that uses it (v_add_f32_dpp, v_max_u32_dpp).
1523  template <int OFF>
1524  DINLINE int lamport_xor_lane_i(int i)
1525  {
1526  #if AITER_GFX1250_HAVE_B128
1527      if constexpr(OFF == 16)
1528          return __builtin_amdgcn_permlanex16(i, i, 0x76543210, 0xfedcba98, false, false);
1529      else
1530          return __builtin_amdgcn_update_dpp(i, i, 0x160 + OFF, 0xf, 0xf, true); // row_xmask:OFF
1531  #else
1532      return __shfl_xor(i, OFF); // other archs only build the kernel, never run it
1533  #endif
1534  }
1535  
1536  template <int OFF>
1537  DINLINE float lamport_xor_lane(float v)
1538  {
1539      return __builtin_bit_cast(float, lamport_xor_lane_i<OFF>(__builtin_bit_cast(int, v)));
1540  }
1541  
1542  // The 32-lane butterfly of block_allreduce_f32 (xor 16, 8, 4, 2, 1), the same
1543  // adds in the same order; every lane gets the sum.
1544  DINLINE float lamport_wave_sum(float p)
1545  {
1546      p = p + lamport_xor_lane<16>(p);
1547      p = p + lamport_xor_lane<8>(p);
1548      p = p + lamport_xor_lane<4>(p);
1549      p = p + lamport_xor_lane<2>(p);
1550      p = p + lamport_xor_lane<1>(p);
1551      return p;
1552  }
1553  
1554  // Max over the wave of non-negative, non-NaN floats as their bits: the bits of
1555  // such floats order as unsigned integers, and an integer max folds into DPP.
1556  DINLINE uint32_t lamport_wave_max_bits(uint32_t p)
1557  {
1558  #define LAMPORT_MAX_STEP(OFF)                                                     \
1559      {                                                                             \
1560          const uint32_t q = (uint32_t)lamport_xor_lane_i<OFF>((int)p);             \
1561          p                = p > q ? p : q;                                         \
1562      }
1563      LAMPORT_MAX_STEP(16) LAMPORT_MAX_STEP(8) LAMPORT_MAX_STEP(4) LAMPORT_MAX_STEP(2)
1564      LAMPORT_MAX_STEP(1)
1565  #undef LAMPORT_MAX_STEP
1566      return p;
1567  }
1568  
1569  // Sum of the row's wave-group partials in group order, as block_allreduce_f32
1570  // folds its waves: thread k of the block owns group threadIdx.x / 32 + k *
1571  // waves of each of its PACKS packs (waves = blockDim.x / 32, passed in: blockDim
1572  // is a kernel argument load). One barrier: after it every wave adds the
1573  // partials itself, so no second barrier hands the result around.
1574  template <int PACKS>
1575  DINLINE float lamport_row_sum(const float (&v)[PACKS], int row_groups, int waves)
1576  {
1577      __shared__ float partial[32];
1578      const int lane = threadIdx.x % kLamportWave;
1579  #pragma unroll
1580      for(int k = 0; k < PACKS; ++k)
1581      {
1582          const float p = lamport_wave_sum(v[k]);
1583          if(lane == 0)
1584              partial[threadIdx.x / kLamportWave + k * waves] = p;
1585      }
1586      lamport_lds_barrier();
1587      // every lane reads the same address (an LDS broadcast) and adds on the
1588      // VALU: a readlane loop would queue every wave of the CU on its SALU
1589      float r = partial[0];
1590      for(int g = 1; g < row_groups; ++g)
1591          r += partial[g];
1592      return r;
1593  }
```

- 1518–1540 `lamport_xor_lane_i<OFF>`（和 float 版的 `lamport_xor_lane`）：取 lane (i xor OFF) 的值，全在寄存器里完成。
  - OFF = 16 用 `permlanex16`：32 个 lane 分两排，每个 lane 换到另一排的同一位置。
  - OFF = 1/2/4/8 用 DPP `row_xmask`：排内 lane i 读 lane i xor OFF。最后一个参数 `bound_ctrl = true` 不改变任何值（row_xmask 读的都是本排存在的 lane），但打开后编译器能把"取值"和后面的加法或求最大值合成一条指令（`v_add_f32_dpp`、`v_max_u32_dpp`）。
  - 原来的 `__shfl_xor` 要经过 LDS 的交换电路（ds_bpermute），每步慢几十到上百个时钟。
- 1542–1552 `lamport_wave_sum`：wave 内 32 个数求和，xor 16、8、4、2、1 依次相加。加法顺序和 car_fused 的 `block_allreduce_f32` 完全一样，所以结果逐位一致。
- 1554–1567 `lamport_wave_max_bits`：wave 内求最大值，按无符号整数比较，只用于 FP8 的 amax（理由见 8.10）。
- 1569–1593 `lamport_row_sum`：整行求和。
  - 每个线程负责 PACKS 个包，各有一个部分和；每个包先做 wave 内求和，lane 0 把结果写到 `partial[第几个 wave + k × waves]`，这个下标正好就是这组包在行里的编号。
  - `lamport_lds_barrier()` 同步一次。
  - 之后每个 lane 从 `partial[0]` 按顺序加到 `partial[row_groups-1]`：必须按组号顺序加（照抄 car_fused），每个 wave 自己从头加一遍，不需要第二次同步。所有 lane 读同一个地址，LDS 会广播，没有冲突。

## 7. kernel 说明和参数
```cpp
1595  // OutT is T, or an FP8 type: then output gets the per-token quant of
1596  // ar_rmsnorm_gfx1250, scale_out a scale a row and bf16_output (unless null)
1597  // the T output. MAX_THREADS only bounds the block; blockDim.x = hidden / 8 /
1598  // PACKS. SPLIT: the first `senders` blocks send send_packs packs each (see the
1599  // comment on top); the row blocks follow.
1600  //
1601  // Scalar and pointer kernel arguments only, the inboxes one by one: the module
1602  // preloads the first 30 dwords of the kernel arguments into SGPRs (32 user
1603  // SGPRs less the kernel argument pointer). Up to 4 GPUs they hold every
1604  // argument the sends and the epilogue read (p4 .. p7 come after them); an
1605  // argument past them is a scalar load where it is used, on the critical path.
1606  template <typename T, typename OutT, int ngpus, int MAX_THREADS, int PACKS, bool SPLIT>
1607  __global__ void __launch_bounds__(MAX_THREADS, 1) lamport_ar_rmsnorm_gfx1250(
1608      const T* __restrict__ input,
1609      const T* __restrict__ residual_inp,
1610      T* __restrict__ residual_out,
1611      OutT* __restrict__ output,
1612      const T* __restrict__ weight,
1613      uint32_t* __restrict__ round_counter, // [2][slot / 32] tx, rx; one a group of 32 packs
1614      int slot,                             // inbox packs per sender
1615      int rank_gemma,                       // rank, and gemma_norm in bit 8
1616      int hidden,
1617      int senders, // SPLIT only
1618      void* p0,
1619      void* p1,
1620      void* p2,
1621      void* p3,
1622      float* __restrict__ scale_out, // [rows], FP8 only
1623      T* __restrict__ bf16_output,   // FP8 only, may be null
1624      float eps,
1625      int send_packs, // SPLIT only, divides the packs of a row
1626      void* p4,       // not preloaded from here on
1627      void* p5,
1628      void* p6,
1629      void* p7)
```

- 模板参数：`OutT` 等于 `T` 是 bf16 输出，FP8 类型是 FP8 输出；`PACKS` 每线程几个包；`SPLIT` 是否 split 模式。`MAX_THREADS` 只是 launch bounds（block 线程数的上限）。
- **参数顺序是故意排的**：wave 启动时，硬件把参数区前 **30 个 dword（120 字节）**直接放进 SGPR（预载）。启动时最多传 32 个用户 SGPR，其中 2 个被参数区地址 `s[0:1]` 占掉。后面的参数用到时要现从显存读，很慢。所以发送和 epilogue 要用的参数都放在前 120 字节：

| 字节偏移 | 参数 | 说明 |
|---|---|---|
| 0–47 | input, residual_inp, residual_out, output, weight, round_counter | |
| 48 | slot | 一格有多少个包 |
| 52 | rank_gemma | 低 8 位 rank，第 8 位 gemma_norm（预载区放不下单独一个参数） |
| 56 | hidden | |
| 60 | senders | split 发送块个数 |
| 64–95 | p0–p3 | 4 张卡收件箱的地址 |
| 96–111 | scale_out, bf16_output | FP8 输出用 |
| 112 | eps | |
| 116 | send_packs | 每个发送块发多少个包 |
| 120 以后 | p4–p7 | 只有 6/8 卡才用 |

- kernel 里也不用 `gridDim`、`blockDim`：它们是编译器加在参数区最后面的隐藏参数，同样要现读，而且用到的地方都在发送之前。

## 8. kernel 主体

### 8.1 开头准备
```cpp
1630  {
1631      // ar_rmsnorm_gfx1250 compiles each square to a rounded multiply and then an
1632      // add; a contracted fma would change the sum of squares (and its bits)
1633      // depending on how the compiler schedules the PACKS loops.
1634  #pragma clang fp contract(off)
1635      static_assert(sizeof(T) == 2, "sentinel 0xffff is only defined for 2-byte dtypes");
1636      static_assert(MAX_THREADS % kLamportWave == 0, "blocks are whole waves");
1637      constexpr bool QUANT    = !std::is_same_v<T, OutT>;
1638      constexpr int pack_size = 16 / sizeof(T);
1639      using P                 = typename opus::vector_t<T, pack_size>;
1640      union pack
1641      {
1642          P data;
1643          lamport_u32x4 i;
1644      };
1645      const int rank        = rank_gemma & 0xff;
1646      const bool gemma_norm = (rank_gemma >> 8) & 1;
1647  
1648      // static indices only: a runtime index into the inboxes is a scalar load
1649      // in series before each remote store
1650      void* const all[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
1651      char* own          = static_cast<char*>(all[0]);
1652  #pragma unroll
1653      for(int g = 1; g < ngpus; ++g)
1654          own = g == rank ? static_cast<char*>(all[g]) : own;
1655  
1656      const int row_packs        = hidden / pack_size;
1657      uint32_t* const rx_counter = round_counter + slot / kLamportWave;
1658      const lamport_u32x4 sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
```

- 1634 `#pragma clang fp contract(off)`：禁止编译器把"乘再加"合成一条 fma。fma 中间不舍入，平方和的最低位会和 car_fused 不同，而且合不合成取决于编译器怎么排指令。
- 1645–1646：从 `rank_gemma` 拆出 rank 和 gemma_norm。
- 1650–1654：`all[8]` 用 8 个标量参数组成；`own = all[rank]` 用编译期展开的循环选出来。如果写成 `all[rank]` 这种运行时下标，每次取地址都要先发一条标量读，会挡在发送前面。
- 1656 `row_packs = hidden / 8`；1657 `rx_counter` 是计数器数组的后半段（前半段是 tx）；1658 `sentinel` 是 16 字节全 0xff，回填用。

### 8.2 split 模式的发送块
```cpp
1660      if constexpr(SPLIT)
1661      {
1662          if((int)blockIdx.x < senders)
1663          {
1664              // a sender block: send_packs packs, one a thread; the other waves
1665              // of the block (it has the row blocks' size) exit
1666              const int t =
1667                  __builtin_amdgcn_readfirstlane(threadIdx.x / kLamportWave) * kLamportWave;
1668              if(t >= send_packs)
1669                  return;
1670              const int p          = blockIdx.x * send_packs + threadIdx.x;
1671              const int wave       = __builtin_amdgcn_readfirstlane(p / kLamportWave);
1672              const uint32_t round = round_counter[wave];
1673              pack v;
1674              v.data = reinterpret_cast<const P*>(input)[p];
1675  #pragma unroll
1676              for(int j = 0; j < 4; ++j)
1677                  v.i[j] = lamportCanonicalize(v.i[j]);
1678  #pragma unroll
1679              for(int g = 0; g < ngpus; ++g)
1680                  if(g != rank)
1681                      lamport_store_sys(lamport_slot(all[g], round, rank, ngpus, slot), p, v.i);
1682              // re-arm buffer (round + 2) % 3 of this pack here, off the row
1683              // blocks: it held the previous call's data, which the previous
1684              // kernel consumed, and a peer is at most one round ahead
1685              const uint32_t rearm = (round + 2) % kLamportBuffers;
1686  #pragma unroll
1687              for(int g = 0; g < ngpus; ++g)
1688                  if(g != rank)
1689                      lamport_store_sys(lamport_slot(own, rearm, g, ngpus, slot), p, sentinel);
1690              if(threadIdx.x % kLamportWave == 0)
1691                  round_counter[wave] = (round + 1) % kLamportBuffers;
1692              return;
1693          }
1694      }
```

- 1662：grid 里前 `senders` 个 block 是发送块。
- 1666–1669：发送块和接收块的线程数一样（一次 launch 只能有一种 blockDim），多出来的 wave 直接退出。
- 1670 `p`：本线程负责的包（全局编号）；1672 读 tx 轮次。
- 1673–1677：读本卡输入的第 p 个包，canonicalize。
- 1678–1681：写到每张别的卡的 `inbox[round][本卡][p]`。
- 1682–1689：**回填**，把自己收件箱 `inbox[(round+2)%3][每张别的卡][p]` 写成 0xffff。可以由发送块来做：这块缓冲是上一轮用过的，上一轮的 kernel 已经结束、数据已经读过；对方最多领先一轮，这一轮不会往这块写。
- 1690–1691：每个 wave 的 lane 0 把 tx 计数器加 1。

### 8.3 接收块负责哪些包（row 模式的 block 也一样）
```cpp
1696      // a row block: thread t owns packs t + k * block of the row
1697      const int block = row_packs / PACKS; // = blockDim.x, without a kernel argument load
1698      const int row   = SPLIT ? (int)blockIdx.x - senders : (int)blockIdx.x;
1699      const int tid0  = row * row_packs + threadIdx.x;
1700      int tid[PACKS];
1701      uint32_t round[PACKS];
1702  #pragma unroll
1703      for(int k = 0; k < PACKS; ++k)
1704      {
1705          tid[k] = tid0 + k * block;
1706          // blocks are whole waves, so this is uniform over the wave
1707          const int wave = __builtin_amdgcn_readfirstlane(tid[k] / kLamportWave);
1708          round[k]       = SPLIT ? rx_counter[wave] : round_counter[wave];
1709      }
```

- 1697 `block = row_packs / PACKS`：block 的线程数，等于 `blockDim.x`，但不用去读参数区。
- 1698 `row`：第几行（split 模式要减去前面的发送块）。
- 1705 `tid[k] = tid0 + k × block`：线程 t 负责本行第 t、t+block、t+2×block … 个包。例：6144 一行 768 个包，PACKS=1 时 768 个线程，每线程 1 个包；PACKS=4 时 192 个线程，线程 t 负责第 t、t+192、t+384、t+576 个包。
- 1708：读这个包所在组的轮次（split 读 rx，row 读 tx）。`readfirstlane` 告诉编译器这个值整个 wave 相同，可以用标量读。

### 8.4 读本卡数据，row 模式发送
```cpp
1711      pack local[PACKS];
1712  #pragma unroll
1713      for(int k = 0; k < PACKS; ++k)
1714      {
1715          local[k].data = reinterpret_cast<const P*>(input)[tid[k]];
1716  #pragma unroll
1717          for(int j = 0; j < 4; ++j)
1718              local[k].i[j] = lamportCanonicalize(local[k].i[j]);
1719      }
1720  
1721      if constexpr(!SPLIT)
1722      {
1723  #pragma unroll
1724          for(int k = 0; k < PACKS; ++k)
1725          {
1726  #pragma unroll
1727              for(int g = 0; g < ngpus; ++g)
1728                  if(g != rank)
1729                      lamport_store_sys(
1730                          lamport_slot(all[g], round[k], rank, ngpus, slot), tid[k], local[k].i);
1731          }
1732          lamport_kernarg_wait();
1733      }
```

- 1711–1719：读本卡输入的包并 canonicalize。求和时本卡这一份直接用它，不经过收件箱。
- 1721–1733：row 模式自己发送，写到每张别的卡的 `inbox[round[k]][本卡][tid[k]]`，发完调用 `lamport_kernarg_wait()`（见第 5 节）。

### 8.5 提前读 residual 和 weight
```cpp
1735      // local operands of the epilogue, loaded before the wait
1736      P res[PACKS], w[PACKS];
1737  #pragma unroll
1738      for(int k = 0; k < PACKS; ++k)
1739      {
1740          res[k] = reinterpret_cast<const P*>(residual_inp)[tid[k]];
1741          w[k]   = reinterpret_cast<const P*>(weight)[tid[k] - row * row_packs];
1742      }
```

- 这两个都在本卡。在等对方数据之前就发出读请求，等待期间它们也在路上，不额外占时间。
- weight 的下标是本行内的列号 `tid[k] - row × row_packs`。

### 8.6 轮询
```cpp
1744      // the peers only, rank + 1 .. rank + ngpus - 1: the slot offsets are
1745      // uniform (SGPR), so this costs no extra address math per load
1746      pack scratch[PACKS][ngpus - 1];
1747      bool pending;
1748      do
1749      {
1750          // without the clobber the loads are hoisted out of the loop
1751          asm volatile("" ::: "memory");
1752  #pragma unroll
1753          for(int k = 0; k < PACKS; ++k)
1754  #pragma unroll
1755              for(int i = 0; i < ngpus - 1; ++i)
1756              {
1757                  const int g     = (rank + 1 + i) % ngpus;
1758                  scratch[k][i].i =
1759                      lamport_load_sys(lamport_slot(own, round[k], g, ngpus, slot), tid[k]);
1760              }
1761          // issue all loads before any mask math
1762          __builtin_amdgcn_sched_barrier(0);
1763          uint32_t m = 0;
1764  #pragma unroll
1765          for(int k = 0; k < PACKS; ++k)
1766  #pragma unroll
1767              for(int i = 0; i < ngpus - 1; ++i)
1768  #pragma unroll
1769                  for(int j = 0; j < 4; ++j)
1770                      m |= lamportSentinelMask(scratch[k][i].i[j]);
1771          pending = m != 0;
1772      } while(pending);
```

- 1746 `scratch[PACKS][ngpus-1]`：存另外 3 张卡发来的包（只读别的卡的格，自己那格永远是 0xffff）。
- 1750 `asm volatile("" ::: "memory")`：告诉编译器内存可能变了，不许把读挪到循环外面。
- 1752–1760：对每个包、每张别的卡 g（rank+1、rank+2、rank+3），读 `inbox[round[k]][g][tid[k]]`。
- 1762 `sched_barrier(0)`：先把这一轮的读全部发出去，再开始检查，让几次读同时在路上。
- 1763–1771：所有读回来的数据算哨兵掩码，还有 0xffff 就继续循环。
- 每个线程只等自己的包，不需要整个 block 一起等。一个 wave 里只要还有一个 lane 没收齐，整个 wave 就继续循环。

### 8.7 求和、加 residual、平方和
```cpp
1774      // same steps as ar_rmsnorm_gfx1250
1775      float x[PACKS][pack_size];
1776      float square_sum[PACKS];
1777  #pragma unroll
1778      for(int k = 0; k < PACKS; ++k)
1779      {
1780  #pragma unroll
1781          for(int j = 0; j < pack_size; ++j)
1782              x[k][j] = 0.f;
1783          // rank order, as ar_rmsnorm_gfx1250: peer g sits at (g - rank - 1) mod ngpus
1784  #pragma unroll
1785          for(int g = 0; g < ngpus; ++g)
1786          {
1787              const int at = (g - rank - 1 + ngpus) % ngpus; // ngpus - 1: this rank
1788              P v          = local[k].data;
1789  #pragma unroll
1790              for(int i = 0; i < ngpus - 1; ++i)
1791                  v = at == i ? scratch[k][i].data : v;
1792  #pragma unroll
1793              for(int j = 0; j < pack_size; ++j)
1794                  x[k][j] += upcast_s(v[j]);
1795          }
1796          P sum;
1797          square_sum[k] = 0.f;
1798  #pragma unroll
1799          for(int j = 0; j < pack_size; ++j)
1800          {
1801              x[k][j] = upcast_s(downcast_s<T>(x[k][j])) + upcast_s(res[k][j]);
1802              sum[j]  = downcast_s<T>(x[k][j]);
1803              square_sum[k] += x[k][j] * x[k][j];
1804          }
1805          reinterpret_cast<P*>(residual_out)[tid[k]] = sum;
1806      }
```

- 1783–1795：按卡号 0、1、2、3 的顺序做 fp32 累加（照抄 car_fused）。`at` 是卡 g 的数据在 `scratch` 里的位置；`at = ngpus-1` 表示本卡，用 `local`。
- 1801 `x = float(bf16(x)) + float(res)`：all-reduce 结果先舍入成 bf16，再在 fp32 里加 residual（原因见 0.1）。
- 1802 `sum = bf16(x)` → 1805 写 `residual_out`。
- 1803 平方和：每个平方先舍入再加（因为 1634 禁止了 fma）。

### 8.8 整行求 inv_rms
```cpp
1808      // FP8: the row amax, merged by an LDS atomic max after the sum barrier
1809      __shared__ uint32_t amax_bits;
1810      if constexpr(QUANT)
1811          if(threadIdx.x == 0)
1812              amax_bits = 0;
1813      const int row_groups = row_packs / kLamportWave;
1814      const float inv_rms =
1815          rsqrtf(lamport_row_sum<PACKS>(square_sum, row_groups, block / kLamportWave) / hidden +
1816                 eps);
```

- 1809–1812：FP8 时由 0 号线程把 LDS 里的 `amax_bits` 清零。必须在下面这次同步之前，后面的原子操作都在同步之后。
- 1814：`inv_rms = rsqrt(整行平方和 / hidden + eps)`，`lamport_row_sum` 里有 block 的第一次同步。

### 8.9 乘 weight，写 bf16 输出
```cpp
1818      P o[PACKS];
1819  #pragma unroll
1820      for(int k = 0; k < PACKS; ++k)
1821      {
1822  #pragma unroll
1823          for(int j = 0; j < pack_size; ++j)
1824          {
1825              const float wj = upcast_s(w[k][j]) + (gemma_norm ? 1.f : 0.f);
1826              o[k][j]        = downcast_s<T>(x[k][j] * (inv_rms * wj));
1827          }
1828      }
1829      if constexpr(!QUANT)
1830      {
1831  #pragma unroll
1832          for(int k = 0; k < PACKS; ++k)
1833              reinterpret_cast<P*>(output)[tid[k]] = o[k];
1834      }
```

- 1825 `wj = float(w) + (gemma_norm ? 1 : 0)`：gemma 版 RMSNorm 用 (1 + weight)。
- 1826 `o = bf16(x × (inv_rms × wj))`：先算 `inv_rms × wj` 再乘 x，顺序照抄 car_fused。
- 1829–1834：bf16 输出直接写 `output`。

### 8.10 FP8 输出
```cpp
1835      else
1836      {
1837          // the steps of ar_rmsnorm_gfx1250 (amax of the T output, scale = amax /
1838          // max, multiply by rcp(scale), clamp, cvt_pk), so the FP8 paths match bit
1839          // for bit. amax starts at 0 and fmaxf drops NaNs: it is never negative
1840          // nor NaN, so its bits order as unsigned integers (lamport_wave_max_bits)
1841          // and the order of the row reduction does not change it.
1842          uint32_t a = 0;
1843  #pragma unroll
1844          for(int k = 0; k < PACKS; ++k)
1845          {
1846              float amax = 0.f;
1847  #pragma unroll
1848              for(int j = 0; j < pack_size; ++j)
1849                  amax = fmaxf(amax, fabsf(upcast_s(o[k][j])));
1850              const uint32_t b = __builtin_bit_cast(uint32_t, amax);
1851              a                = a > b ? a : b;
1852              if(bf16_output != nullptr)
1853                  reinterpret_cast<P*>(bf16_output)[tid[k]] = o[k];
1854          }
1855          a = lamport_wave_max_bits(a);
1856          if(threadIdx.x % kLamportWave == 0)
1857              __hip_atomic_fetch_max(
1858                  &amax_bits, a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
1859          lamport_lds_barrier();
1860          const float row_amax = __builtin_bit_cast(float, amax_bits);
1861          const float fp8_max  = opus::cast<opus::fp32_t>(opus::numeric_limits<OutT>::max());
1862          const float scale    = row_amax * (float)(1. / fp8_max);
1863          const float inv      = __builtin_amdgcn_rcpf(scale);
1864  #pragma unroll
1865          for(int k = 0; k < PACKS; ++k)
1866          {
1867              opus::vector_t<OutT, pack_size> q;
1868  #pragma unroll
1869              for(int j = 0; j < pack_size; j += 2)
1870              {
1871                  const float qa =
1872                      __builtin_amdgcn_fmed3f(upcast_s(o[k][j]) * inv, -fp8_max, fp8_max);
1873                  const float qb =
1874                      __builtin_amdgcn_fmed3f(upcast_s(o[k][j + 1]) * inv, -fp8_max, fp8_max);
1875                  reinterpret_cast<uint16_t*>(&q)[j / 2] =
1876                      __builtin_amdgcn_cvt_pk_fp8_f32(qa, qb, 0, false);
1877              }
1878              reinterpret_cast<opus::vector_t<OutT, pack_size>*>(output)[tid[k]] = q;
1879          }
1880          if(threadIdx.x == 0)
1881              scale_out[row] = scale;
1882      }
```

- 1844–1854：每个包求 `amax = max |o|`（o 已经是舍入成 bf16 的输出），本线程几个包的 amax 再取最大值；如果 `bf16_output` 不为空，同时写一份 bf16 输出。
- **为什么能按整数比较**：amax 从 0 开始，取的都是绝对值，`fmaxf` 又会丢掉 NaN，所以一定是非负、非 NaN 的 float。这种 float 的位模式当成无符号整数比大小，结果和按 float 比一样。整数比较能直接合进 DPP，不需要 NaN 规范化。
- 1855：wave 内最大值。
- 1856–1858：每个 wave 的 lane 0 用 `ds_max_u32`（LDS 原子最大值）把结果合进 `amax_bits`。
- 1859：block 第二次同步。1860：每个线程只读 `amax_bits` 这一个数，不用再做一次规约。
- 1861–1863：`scale = row_amax / 448`（FP8 最大值），`inv = rcp(scale)`（硬件近似倒数，和 car_fused 一样）。
- 1864–1879：`o × inv`，用 `fmed3` 一条指令 clamp 到 ±448，两个一组转成 FP8，8 个 FP8 一起写 `output`。
- 1880–1881：每行由 0 号线程写 `scale_out[row]`。

### 8.11 回填和更新轮次
```cpp
1884      if constexpr(!SPLIT)
1885      {
1886  #pragma unroll
1887          for(int k = 0; k < PACKS; ++k)
1888          {
1889              const uint32_t rearm = (round[k] + 2) % kLamportBuffers;
1890  #pragma unroll
1891              for(int g = 0; g < ngpus; ++g)
1892                  if(g != rank)
1893                      lamport_store_sys(lamport_slot(own, rearm, g, ngpus, slot), tid[k], sentinel);
1894          }
1895      }
1896  
1897      if(threadIdx.x % kLamportWave == 0)
1898      {
1899  #pragma unroll
1900          for(int k = 0; k < PACKS; ++k)
1901          {
1902              const uint32_t next = (round[k] + 1) % kLamportBuffers;
1903              if constexpr(!SPLIT)
1904                  round_counter[tid[k] / kLamportWave] = next;
1905              rx_counter[tid[k] / kLamportWave] = next;
1906          }
1907      }
1908  }
```

- 1884–1895：row 模式自己回填（split 模式已经由发送块回填），把 `inbox[(round[k]+2)%3][每张别的卡][tid[k]]` 写成 0xffff。
- 1897–1907：每个 wave 的 lane 0 更新轮次：row 模式 tx、rx 都加 1；split 模式只加 rx（tx 由发送块加）。

---

## 9. host 端：`LamportArRms`

### 9.1 结构体
```cpp
2480  // The Lamport inbox of a communicator: one per CustomAllreduce, any row width
2481  // and row count up to slot packs (see lamport_ar_rmsnorm_gfx1250).
2482  struct LamportArRms
2483  {
2484      LamportPeers peers{};
2485      uint32_t* d_round = nullptr; // [2][slot / 32]: tx, then rx counters; rank-local
2486      int slot          = 0;       // the most packs a call takes
2487      int rank          = 0;
2488      int world_size    = 0;
2489  
2490      ~LamportArRms()
2491      {
2492          if(d_round)
2493              (void)hipFree(d_round);
2494      }
```

- `peers`：每张卡收件箱的地址；`d_round`：轮次计数器，`[2][slot/32]`，前半 tx、后半 rx；`slot`：一格有多少个包。

### 9.2 每线程几个包：`row_block_packs`
```cpp
2496      // Packs a thread of a row block: the fewest that keep the block at most 256
2497      // threads (128 from 257 rows, 64 from 513: more, smaller blocks for big
2498      // calls) and divide the row's groups of 32 packs (4x gfx1250, TP2/TP4,
2499      // hidden 1024-8192, M 1-2048).
2500      static int row_block_packs(int row_packs, int64_t rows)
2501      {
2502          const int most   = rows <= 256 ? 256 : rows <= 512 ? 128 : 64;
2503          const int groups = row_packs / kLamportWave;
2504          int packs        = 1;
2505          for(int p : {1, 2, 4, 8})
2506          {
2507              if(groups % p != 0)
2508                  continue;
2509              packs = p;
2510              if(row_packs / p <= most)
2511                  break;
2512          }
2513          return packs;
2514      }
```

- 行数越多，block 越小、个数越多，实测更快：`most` 是 block 线程数上限，rows ≤ 256 时 256，≤ 512 时 128，再多是 64。
- 在 1、2、4、8 里，只考虑能整除"每行组数"的值，选让 block 线程数 ≤ `most` 的最小值；都不满足就取能整除的最大值。

### 9.3 `run()`：选模式并启动 kernel
```cpp
2516      // OutT: T, or an FP8 type with scale_out (and bf16_out unless null), see
2517      // lamport_ar_rmsnorm_gfx1250
2518      template <typename T, typename OutT>
2519      void run(hipStream_t stream,
2520               const T* input,
2521               const T* residual_inp,
2522               T* residual_out,
2523               OutT* output,
2524               const T* weight,
2525               float* scale_out,
2526               T* bf16_out,
2527               float eps,
2528               int64_t rows,
2529               int hidden,
2530               bool gemma_norm)
2531      {
2532          constexpr int pack_size = 16 / sizeof(T);
2533          if(hidden <= 0 || hidden % (pack_size * kLamportWave) != 0 ||
2534             hidden / pack_size > kLamportRowMaxThreads)
2535              throw std::runtime_error("lamport allreduce rmsnorm: hidden must be a multiple of " +
2536                                       std::to_string(pack_size * kLamportWave) + " and at most " +
2537                                       std::to_string(pack_size * kLamportRowMaxThreads) +
2538                                       ", got " + std::to_string(hidden));
2539          const int row_packs = hidden / pack_size;
2540          const int64_t packs = rows * row_packs;
2541          if(packs > slot)
2542              throw std::runtime_error("lamport allreduce rmsnorm: " + std::to_string(packs * 16) +
2543                                       "B exceeds the inbox cap of " +
2544                                       std::to_string((int64_t)slot * 16) + "B");
2545          if(packs == 0)
2546              return;
2547          // Split (sender blocks, then a row block of one pack a thread) while the
2548          // call is small: one CU sending and polling a whole row is the limit
2549          // there. Rows of up to 1024 values stay in row mode. 4x gfx1250, bf16
2550          // and FP8 out: TP4 up to 64 rows, TP2 up to 32K packs (rows x hidden / 8).
2551          const bool split =
2552              row_packs > 128 && (world_size >= 4 ? rows <= 64 : packs <= 32768);
2553          const int packs_per_thread = split ? 1 : row_block_packs(row_packs, rows);
2554          const int row_threads      = row_packs / packs_per_thread;
2555          // sender blocks of 256 packs, a multiple of 32 that divides the row, so
2556          // that no sender block runs past the call
2557          int send_packs = std::min(256, row_packs);
2558          while(row_packs % send_packs != 0)
2559              send_packs -= kLamportWave;
2560          const int senders = split ? (int)(packs / send_packs) : 0;
2561  
2562  #define LAMPORT_RMS_ARGS                                                                    \
2563      input, residual_inp, residual_out, output, weight, d_round, slot,                       \
2564          rank | (gemma_norm ? 0x100 : 0), hidden, senders, peers.ptrs[0], peers.ptrs[1],     \
2565          peers.ptrs[2], peers.ptrs[3], scale_out, bf16_out, eps, send_packs, peers.ptrs[4],  \
2566          peers.ptrs[5], peers.ptrs[6], peers.ptrs[7]
2567  #define LAUNCH_LAMPORT_RMS_PK(NG, PK, SPLIT)                                                \
2568      lamport_ar_rmsnorm_gfx1250<T, OutT, NG, (kLamportRowMaxThreads / PK + 31) / 32 * 32, PK, \
2569                                 SPLIT>                                                       \
2570          <<<senders + (int)rows, row_threads, 0, stream>>>(LAMPORT_RMS_ARGS)
2571  #define LAUNCH_LAMPORT_RMS(NG)                                                              \
2572      if(split)                                                                               \
2573          LAUNCH_LAMPORT_RMS_PK(NG, 1, true);                                                 \
2574      else                                                                                    \
2575          switch(packs_per_thread)                                                            \
2576          {                                                                                   \
2577          case 2: LAUNCH_LAMPORT_RMS_PK(NG, 2, false); break;                                 \
2578          case 4: LAUNCH_LAMPORT_RMS_PK(NG, 4, false); break;                                 \
2579          case 8: LAUNCH_LAMPORT_RMS_PK(NG, 8, false); break;                                 \
2580          default: LAUNCH_LAMPORT_RMS_PK(NG, 1, false); break;                                \
2581          }
2582          DISPATCH_NGPUS_1250(world_size, LAUNCH_LAMPORT_RMS);
2583  #undef LAUNCH_LAMPORT_RMS
2584  #undef LAUNCH_LAMPORT_RMS_PK
2585  #undef LAMPORT_RMS_ARGS
2586      }
```

- 2533–2546：检查 hidden 是 256 的倍数且 ≤ 8192，总包数不超过 slot。
- 2547–2552 **选模式**：hidden > 1024（一行超过 128 个包）时，TP ≥ 4 且 rows ≤ 64，或者 TP2 且总包数 ≤ 32768，就用 split；其他用 row。
- 2553–2554：split 时每线程 1 个包，row 时用 `row_block_packs`；`row_threads` 是接收块（或 row 模式 block）的线程数。
- 2555–2560：`send_packs` 从 256 起往下减 32，直到能整除每行包数，这样最后一个发送块不会越界，kernel 里也不需要总包数；`senders = 总包数 / send_packs`。
- 2562–2566 `LAMPORT_RMS_ARGS`：按第 7 节的顺序传参，`rank | (gemma ? 0x100 : 0)`。
- 2567–2570 `LAUNCH_LAMPORT_RMS_PK`：grid = `senders + rows`（row 模式 senders = 0），block = `row_threads`，`MAX_THREADS = 1024 / PK`。
- 2571–2581 `LAUNCH_LAMPORT_RMS`：split 用 PK = 1 的 split 实例，row 按每线程包数选 1/2/4/8 的实例。
- 2582 `DISPATCH_NGPUS_1250`：按卡数 2/4/6/8 选模板实例。
- 选择规则来自实测（TP4/TP2，hidden 1024~8192，M 1~2048，bf16 和 FP8，6 种固定配置逐个测）：自动选择比"每个形状都挑最优"平均只慢 0.1%（TP4）和 1.4%（TP2）。

## 10. 初始化：`init_lamport_ar_rms`
```cpp
3518      // -----------------------------------------------------------------------
3519      // Lamport fused all-reduce + RMSNorm, see lamport_ar_rmsnorm_gfx1250.
3520      //
3521      // all_ptrs[i] is rank i's buffer of lamport_buffer_bytes(world_size_,
3522      // max_bytes / 16) bytes, mapped into this process (all_ptrs[rank_] is this
3523      // rank's own); max_bytes is the largest input a call takes. Once per
3524      // communicator. The caller must barrier across ranks after this returns,
3525      // otherwise a fast rank's first push can be wiped by a slow rank's 0xff
3526      // memset.
3527      // -----------------------------------------------------------------------
3528      void init_lamport_ar_rms(const std::vector<int64_t>& all_ptrs, int64_t max_bytes)
3529      {
3530          if(lamport_)
3531              throw std::runtime_error("init_lamport_ar_rms: already initialized");
3532          if(world_size_ > kMaxNgpusAr || (int)all_ptrs.size() != world_size_)
3533              throw std::invalid_argument(
3534                  "init_lamport_ar_rms: need one buffer per rank, at most 8 ranks");
3535          // one round counter and one partial a wave, so whole waves of packs
3536          if(max_bytes <= 0 || max_bytes % (16 * kLamportWave) != 0 ||
3537             max_bytes / 16 > std::numeric_limits<int>::max() / (kLamportBuffers * kMaxNgpusAr))
3538              throw std::invalid_argument("init_lamport_ar_rms: max_bytes must be a positive "
3539                                          "multiple of 512");
3540          const int64_t slot = max_bytes / 16;
3541          auto state         = std::make_unique<LamportArRms>();
3542          state->rank        = rank_;
3543          state->world_size  = world_size_;
3544          state->slot        = (int)slot;
3545          for(int i = 0; i < world_size_; i++)
3546              state->peers.ptrs[i] = reinterpret_cast<void*>(all_ptrs[i]);
3547          HIP_CALL(hipMemset(
3548              state->peers.ptrs[rank_], 0xff, lamport_ar_rms_buffer_bytes(world_size_, slot)));
3549          HIP_CALL(hipMemset(static_cast<char*>(state->peers.ptrs[rank_]) +
3550                                 lamport_ar_rms_buffer_bytes(world_size_, slot),
3551                             0,
3552                             lamport_escape_bytes(world_size_, slot)));
3553          HIP_CALL(hipMemset(static_cast<char*>(state->peers.ptrs[rank_]) +
3554                                 lamport_two_shot_offset(world_size_, slot),
3555                             0xff,
3556                             lamport_two_shot_bytes(world_size_, slot)));
3557          // tx counters, then rx counters (see lamport_ar_rmsnorm_gfx1250 SPLIT)
3558          HIP_CALL(hipMalloc((void**)&state->d_round, sizeof(uint32_t) * 2 * (slot / kLamportWave)));
3559          HIP_CALL(hipMemset(state->d_round, 0, sizeof(uint32_t) * 2 * (slot / kLamportWave)));
3560          HIP_CALL(hipDeviceSynchronize());
3561          lamport_ = std::move(state);
3562      }
```

- 3536–3539：`max_bytes / 16`（一格的包数）乘 3 × 8 不能超过 int 的范围，所以一格远小于 2 GiB，第 4 节 buffer 指令的 32 位偏移一定够用。
- 3547–3556：自己收件箱的数据区填 0xff（哨兵），escape 区填 0，two-shot 区填 0xff。
- 3557–3559：计数器分配 `2 × slot/32` 个并清零（tx、rx 各一半）。

## 11. 其他 Lamport kernel：同时写 rx 计数器

1-shot（含 AR+add）、2-shot、all-gather 三个 kernel 结束时，除了 tx 也写 rx，否则它们和 split 模式的 RMS 交替调用时，接收块会读到旧轮次。三处写法相同，以 1-shot 为例：
```cpp
2041      if(threadIdx.x % kLamportWave == 0)
2042      {
2043          round_counter[wave]                       = (round + 1) % kLamportBuffers;
2044          // and rx: a split lamport_ar_rmsnorm_gfx1250 call reads the rx counters
2045          round_counter[slot / kLamportWave + wave] = (round + 1) % kLamportBuffers;
2046      }
2047  }
```
另外两处在 2235–2241（2-shot）和 2472–2478（all-gather）。

