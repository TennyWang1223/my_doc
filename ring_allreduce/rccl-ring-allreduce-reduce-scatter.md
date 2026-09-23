# RCCL Ring AllReduce · Reduce-Scatter 逐步图解(TP=4)

环境:gfx1250,单机 4 卡,环形拓扑 `rank0 → rank1 → rank2 → rank3 →(回绕)rank0`。

## 0. 拓扑 + 输入切分标准

```
                     rank0 ──────▶ rank1 ──────▶ rank2 ──────▶ rank3
                       ▲                                          │
                       └──────────────────────────────────────────┘
                                  (环形回绕,4 卡一圈)

切分标准: chunk_size = input_size / 4,偏移量固定:
    offset(C0)=0            offset(C1)=chunk_size        offset(C2)=2*chunk_size      offset(C3)=3*chunk_size
    ├───────────────┼───────────────┼───────────────┼───────────────┤
    0            chunk_size     2*chunk_size     3*chunk_size    input_size
         C0              C1              C2              C3
```

## 1. sendbuff(只读原始输入,全程不变)

```
                C0      C1      C2      C3
rank0.sendbuff:  1       2       3       4
rank1.sendbuff: 10      20      30      40
rank2.sendbuff:100     200     300     400
rank3.sendbuff:1000   2000    3000    4000
```

recvbuff(唯一会被读写的 scratch,4 卡一开始全是垃圾,下面用 `·` 表示):

```
rank0.recvbuff: [ · | · | · | · ]
rank1.recvbuff: [ · | · | · | · ]
rank2.recvbuff: [ · | · | · | · ]
rank3.recvbuff: [ · | · | · | · ]
```

信号量 buffer:每条环边(`rank_i → rank_{i+1}`)各自独立一份,物理上开在**接收方**显存里:

```
        tail(接收方内存,发送方IPC写)   head(发送方内存,接收方IPC写,控制发送节奏)
edge0→1:      tail01 = 0                     head01 = 0
edge1→2:      tail12 = 0                     head12 = 0
edge2→3:      tail23 = 0                     head23 = 0
edge3→0:      tail30 = 0                     head30 = 0
```

---

## Step 1 —— 起始发送(纯搬运,不加法)

```
rank0 ──[C3=4]──────▶ rank1 ──[C0=10]─────▶ rank2 ──[C1=200]────▶ rank3
  ▲                                                                  │
  └───────────────────────[C2=3000]─────────────────────────────────┘

发完之后,每条边立刻做:
  发送方: P2P 写完数据 → __threadfence_system() → 原子 store tail += 1   (release)
  接收方:                                          忙轮询原子 load tail (acquire) 直到看到新值
```

```
tail01=1  tail12=1  tail23=1  tail30=1        ← 4 条边各自的信号量都推进到 1
```

recvbuff(Step1 结束):

```
rank0.recvbuff: [   ·  |   ·  | 3000 |   ·  ]
rank1.recvbuff: [   ·  |   ·  |   ·  |   4  ]
rank2.recvbuff: [  10  |   ·  |   ·  |   ·  ]
rank3.recvbuff: [   ·  |  200 |   ·  |   ·  ]
```

---

## Step 2 —— 第一次归约(本地值 + 刚收到的值 → 转发)

```
rank0 ──[C2=3+3000=3003]───▶ rank1 ──[C3=40+4=44]───▶ rank2 ──[C0=100+10=110]──▶ rank3
  ▲                                                                                 │
  └────────────────────────[C1=2000+200=2200]─────────────────────────────────────┘

每个 rank 这一步做的事完全一样(以 rank0 为例):
  1. 等 tail30 ≥ 1(轮询,已经在 Step1 末尾满足)
  2. 读 rank0.recvbuff[C2]=3000  +  读 rank0.sendbuff[C2]=3  →  算出 3003
  3. P2P 把 3003 写进 rank1.recvbuff[C2]
  4. threadfence_system() → 原子 store tail01 += 1
```

```
tail01=2  tail12=2  tail23=2  tail30=2
```

recvbuff(Step2 结束,`~x~`标记用过的旧值,不会再被读):

```
rank0.recvbuff: [   ·  | 2200 |~3000~|   ·  ]     ← C2 这格 Step1 写的值,Step2 已用完
rank1.recvbuff: [   ·  |   ·  | 3003 | ~4~  ]
rank2.recvbuff: [ ~10~ |   ·  |   ·  |  44  ]
rank3.recvbuff: [  110 |~200~ |   ·  |   ·  ]
```

---

## Step 3 —— 第二次归约(此时值已含 3 个 rank 的贡献)

```
rank0 ──[C1=2+2200=2202]───▶ rank1 ──[C2=30+3003=3033]──▶ rank2 ──[C3=400+44=444]──▶ rank3
  ▲                                                                                     │
  └───────────────────────[C0=1000+110=1110]──────────────────────────────────────────┘
```

```
tail01=3  tail12=3  tail23=3  tail30=3
```

recvbuff(Step3 结束):

```
rank0.recvbuff: [ 1110 |~2200~|~3000~|   ·  ]
rank1.recvbuff: [   ·  | 2202 |~3003~| ~4~  ]
rank2.recvbuff: [ ~10~ |   ·  | 3033 | ~44~ ]
rank3.recvbuff: [~110~ |~200~ |   ·  |  444 ]
```

---

## Step 4 —— 最终归约(本地落地,不再需要跨卡等待)

这一轮每个 rank 只读**自己**的 `sendbuff[i]` + `recvbuff[i]`(Step3 已经就绪,不用等新信号),算完直接原地存回:

```
rank0: sendbuff0[C0]=1    + recvbuff0[C0]=1110 → C0 = 1111  ✔ (存回 rank0.recvbuff[C0])
rank1: sendbuff1[C1]=20   + recvbuff1[C1]=2202 → C1 = 2222  ✔ (存回 rank1.recvbuff[C1])
rank2: sendbuff2[C2]=300  + recvbuff2[C2]=3033 → C2 = 3333  ✔ (存回 rank2.recvbuff[C2])
rank3: sendbuff3[C3]=4000 + recvbuff3[C3]=444  → C3 = 4444  ✔ (存回 rank3.recvbuff[C3])

           ┌───────┐        ┌───────┐        ┌───────┐        ┌───────┐
           │ rank0 │        │ rank1 │        │ rank2 │        │ rank3 │
           │C0=1111│        │C1=2222│        │C2=3333│        │C3=4444│
           │  ✔    │        │  ✔    │        │  ✔    │        │  ✔    │
           └───────┘        └───────┘        └───────┘        └───────┘
           (这一步之后同样的写操作还会顺手转发给下一个 rank,作为 AllGather 的第一跳——
            这部分本文档不展开)
```

recvbuff(Reduce-Scatter 最终状态,对角线 = 全局归约完成):

```
rank0.recvbuff: [ 1111✔ |~2200~|~3000~|   ·  ]
rank1.recvbuff: [   ·   | 2222✔|~3003~| ~4~  ]
rank2.recvbuff: [ ~10~  |   ·  | 3333✔| ~44~ ]
rank3.recvbuff: [~110~  |~200~ |   ·  | 4444✔]
```

核对:`1+10+100+1000=1111` ✔ / `2+20+200+2000=2222` ✔ / `3+30+300+3000=3333` ✔ / `4+40+400+4000=4444` ✔,全部对上。

## 同步方式小结

全程没有 CAS,只有"原子 store(发送方,配 `__threadfence_system()` 当 release)+ 忙轮询原子 load(接收方,acquire)"这一套 tail/head 计数器握手,4 条环边各自独立一份,互不干扰,同一个 rank 的每一轮之间靠代码顺序执行天然衔接。
