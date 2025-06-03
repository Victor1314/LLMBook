# 注意力机制的进阶思考

**元**

本质上是XXX

**反**

多头注意力机制的计算量过大。  



**空**

- MHA，ＭＱＡ，GQA对比:

MHA ，多头注意力机制，transformer论文中提出。

ＭＱＡ， MHA 改进版，每个头有一个Q，所有的头共享KV矩阵。

GQA，　综合版，每个头有一个Q，有多组的KV。

| 机制 | Query    | Key/Value  | 参数量/速度   | 表达能力 |
| ---- | -------- | ---------- | ------------- | -------- |
| MHA  | 每头独立 | 每头独立   | 参数最多/慢   | 最强     |
| MQA  | 每头独立 | 全部头共享 | 参数最少/最快 | 较弱     |
| GQA  | 每头独立 | 每组头共享 | 参数适中/较快 | 适中     |



- FlashAttention是什么？

flashAttention是一种高效的注意力机制实现方式。 传统的注意力计算的方法, 需要计算QK(T)矩阵相乘，时间复杂度为O(n2), 再对大矩阵进行softmax.  flashAttention的核心为，将大矩阵分成block，以block作为核心计算矩阵乘法运算，以及归一化计算。有了flashAttention之后，可以提高计算的效率。

```
┌──────────────┐
│   Q, K, V    │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 分块加载小块 │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 计算分数     │
│ 边softmax    │
│ 边加权求和   │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 处理下一个块 │
└──────────────┘

```



- KVcache的原理

大模型一次吐出一个词。在原始的transformer中，在推理的时候，需要计算重新计算历史token的KV，新的token要生成KVQ，用Q去查询历史token的KV，捕捉特征。 因此历史的KV完全可以复用 。

kv-cache 的 shape 是

[batch_size, n_heads, past_seq_len, head_dim]

past_seq_len：历史token长度，会随着生成步数增长。 

kv-cache面临内存增长的问题，目前有的办法是将模型分到不同的GPU上，或者采用量化的方式减少显存使用。

| 名称       | 维度（多头）                             | 用途              |
| ---------- | ---------------------------------------- | ----------------- |
| Q          | [batch, n_heads, seq_len, head_dim]      | 当前token查询向量 |
| K          | [batch, n_heads, seq_len, head_dim]      | 历史token键向量   |
| V          | [batch, n_heads, seq_len, head_dim]      | 历史token值向量   |
| kv-cache K | [batch, n_heads, past_seq_len, head_dim] | 缓存历史K         |
| kv-cache V | [batch, n_heads, past_seq_len, head_dim] | 缓存历史V         |





- VLLM的加速原理

pageAttention 和 Continuous Batching

传统的kv-cache在进行管理的时候，会分配一个大张量，张量的size为(layer, batch_size, sequence_length, heaer_count,  head_dim)

sequence_length 通常会padding到同一个长度，导致显存浪费。并且分配大张量，会导致显存碎片。

你猜到了吗？ pageAttention 和内存的分页管理思想一样, 不再给请求分配一个大块显存。而是给先将显存分割成较小的页，同一个请求可以横跨不同的页。多个请求的kv可以拼接在一个大张量里。这样不会造成padding浪费。

当请求过来，通过**页表**查询历史token在那一页。

pageAttention,  kv进行矩阵运算的时候，是不是要copy一份? 　并不是。

 在做Attention计算时，**通过高效的索引（gather/scatter）操作，把需要的kv在计算时“逻辑上拼成”一个连续矩阵**

**Continuous Batching**  , 能够将不同时间发起来的请求，拼接到一个batch里推理,GPU利用率更高，相应更快。

也就是能动态的扩展当前batch 



