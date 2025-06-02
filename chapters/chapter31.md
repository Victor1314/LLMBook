# 试验一下
torch.manual_seed(123)

batch_size, block_size, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```



### 注意力机制的进阶思考 

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



### 小结





## 四、实现一个GPT模型

### 最核心模块— transformer block



ransformer block 是大模型的核心组件。模型参数中的层数`num_layers`，指的就是transformer block的个数。gpt2的`num_layers`为48，gpt3为96.

 它包括因果注意力机制和前馈神经网络, 以及在进入因果注意力和前馈神经网络层之前的层归一化。并且通过快捷连接来连接因果注意力层和前馈神经网络层。 

当输入X进来，transformer block如何运转呢？  X会依次经过因果注意力机制和前馈神经网络层的处理。



首先，X会经过被归一化，处理成均值为0，方差为1的向量值 Ynorm。

接着，Ynorm注入因果注意力层，得到Yatt, 然后经过Dropout 得到Yatt_drop

然后应用快捷连接，将 X + Yatt_drop 得到真正的Y。

 

Y作为新的X, 传入反馈神经网络。

同样，X会被归一化处理，处理成均值为0，方差为1的向量值 Ynorm。

接着，经过反馈神经网络，得到Yffn, 并经过Dropout 得到Yffn_drop。

最后再拼接 X 和Yffn_drop 得到Ｙ。　



todo：使用图来描述更加清晰  



```
class TransformerBlock(nn.Module):  # 需要继承nn.Module
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiAtention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            n_heads = cfg["n_heads"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.norm1 = NormLayer(cfg["emb_dim"])
        self.norm2 = NormLayer(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])  
        self.ffn = ForwardFeedLayer(cfg["drop_rate"])  
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)  
        x = self.att(x)    
        x = self.drop(x)
        x = shortcut + x
        
        shortcut = x
        x = self.norm2(x)  
        x = self.ffn(x)
        x = self.drop(x)
        
        return shortcut + x

		
```



#### 多头注意力机制　

#### 前馈神经网络

- 线性层
- 激活函数GELU

前馈神经网络是什么?



神经网络一种模仿生物神经系统的模型。  生物大脑，接受外界的输入，经过神经元的处理，输出电信号，再传递给下一个神经元。经过数千万个神经元的协调，最终输出。



神经元是神经网络的基本单元。作为数学模型，神经元由权重、偏置、激活函数构成。

经由权重、偏置，对输入进行线性变换。

经由激活函数进行非线性变化后输出。  



最基础的神经网络由线性层和激活函数构成 。 线性层是只能进行线性变换的组件。激活函数对线性层的输出进行非线性的变换。

如下定义了一个线性层，din代表输入参数维度，dout代表输出的维度。

```　python
layer = torch.nn.Linear(100,200)  #din,dout
```



举个例子：

一个由两个线性层和激活函数构成的前馈神经网络神经网络

```python
class Feedforward(nn.Module):
   
   def  __init__(self,cfg):
       super().__init__()
       self.layers = nn.Sequential(
       		nn.Linear(cfg["emb_dim"], 4* cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4* cfg[ "emb_dim"], cfg["emb_dim"])
       )
    
    def forward(self,x):
        return self.layers(x)
    

```

思考：假设 cfg["emb_dim"] = 256, 上面这个前馈神经网络有多少个参数？参数中的权重矩阵参数和偏置项分别是多少？

提醒：偏置项个数等于输出维度。

让我们来实现GPT中的前馈神经网络　



GELU函数

```
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```



完整的前馈神经网络

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)
```



#### 层归一化



层归一化是一种增加模型训练稳定性的深度学习技术， 能避训练过程中的梯度消失或者梯度爆炸。

层归一化的流程为，调整神经网络层的输出，让它符合：”均值为0，方差为1“的规则。

再将层归一化之后的输出作为下一个神经网络层的输入。

具体步骤为：

- 求输出的均值和方差
- 输出中各个维度的值减去均值 ； 除以方差的平方根

在transformer block中，两次运用了层归一化。分别在因果注意力机制输入之前，前馈神经网络输入之前。 



下面层归一化层的实现

```
 class Normlayer(nn.Module):
 	def __init__(self,emb_dim):
 		super().__init__()
 		self.eps = 1e-5
 		self.scale = torch.nn.Parameter(torch.ones(emb_dim))
 		self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
 	def forward(self,x):
		mean = x.mean(-1,keepdim= True)
		var  = x.var(-1,keepdim = True,unbiased=False)
		x = (x -mean)/torch.sqrt(var+ self.eps)
		return  self.scale *x + self.shift
 	
```

unbiased=False 的意思是使用有偏方差，原因在于数据量比较大，误差可以忽略不记。

#### 残差连接

快捷连接(shortcut connection，又叫残差连接)是什么？

快捷连接一种在不同层之间增加连接的深度学习技术。快捷解决了反向传播时的梯度过小的问题 ——越靠前的神经网络层梯度越小。

如何实现快捷连接？ 

将当前神经网络的输入 x, 添加到输出之中。也就是x + 输出 = 新的输出。

当你在构建一个多层神经网络时，可以在不同的神经网络层之间添加快捷连接，从而避免梯度消失的问题。

在shortcut connect之前，可以执行dropout，这也是transformer block中的做法。



#### tranformer block代码实现 



```python
from previous_chapters import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块中的Shortcut连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        # 前馈块中的Shortcut连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        return x
```



### GPT类实现（数据输入，输出）



#### GPT核心有哪些组件？

```
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```



logits的 形状: (batch_size, seq_length, vocab_size)



forward函数

- embedding + pos- embedding 
- N个 transformer
- final-norm
- head-out



如何计算模型的参数量 

```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```



### 简单文本生成

使用贪婪解码生成文本。

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过了支持的长度，就对当前上下文进行截断
        # 例如，如果LLM只支持5个token，而上下文长度为10，
        # 那么只有最后5个token会被用作上下文

        idx_cond = idx[:, -context_size:]
        
        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]  

        # 通过softmax函数获得对应的概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 获取概率值最高的单词索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引添加到当前运行的上下文索引序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```







模型是如何结束输出的？ 

​     模型什么时候终止，受到三个因素的影响。max_sequence , max_new_token, 模型主动输出停止符 `eos` .

在推理中，如果序列的长度达到了max_sequence， 模型就不会再输出。具体来说，

**（1）如果达到max_sequence_length（最大序列长度）**

- 这条序列**不能再生成新的token**，直接停止生成。
- 在实际实现中，这条序列会被**mask/pad**，即后续生成步骤中，这条序列不再被“激活”，只保留已生成内容。

**（2）如果提前遇到终止符（如EOS）**

- 这条序列也会停止生成，后续步骤用padding填充。







### GPT架构的进阶思考 

**元**

**反**



Logits每次都需要计算所有，导致会又很多浪费。因此 kv -cache诞生了

**空**



**什么是MOE模型？**
MOE，混合专家模型，是一种神经网络模型。由门控和多个专家网络构成。用于替换前馈神经网络。前馈神经网络是全部参数激活的，也叫做dense模型。而MOE则是先由门控来控制激活哪些专家，可以减少推理时的参数激活量。但MOE模型在训练难度上比Dense模型要高。 



**RMSＮorm是什么？** 

RMSNorm是一种基于均方根的归一化方法。GPT最开始使用的LayerNorm需要先减去均值，再除以方差，使得向量变成均值为０，方差为１的向量。RMSNorm不需要计算均值，也不使用方差来归一化。而是先将计算各项的平方和的均值，开根号，得到均方根。再用各项除以均方根，得到归一化后的值。

RMSＮorm的计算量比常规的LayerNorm要少，逐渐被主流的大模型采用，比如Qwen





### 小结



## 五、如何训练模型



token的编码和解码。

```python
import tiktoken
from previous_chapters import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 增加batch维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 去掉batch维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```





### 如何评估模型的输出？ 



使用输出结果和目标的距离来衡量。  





#### 交叉熵



如何计算交叉熵？  

- 目标词元的概率

  - logits
  - target-token

  ```
  inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                         [40,    1107, 588]])   #  "I really like"]
  
  targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                          [588,  428,  11311]]) #  " really like chocolate"]
  ```

  

  ```
  with torch.no_grad():
      logits = model(inputs)
  
  probas = torch.softmax(logits, dim=-1) # 词表中每个标记的预测概率
  print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
  ```

  

  ```
  batch_idx = 0
  target_probas_1 = probas[batch_idx, [0, 1, 2], targets[batch_idx]]
  print("Batch 1:", target_probas_1)
  
  batch_idx = 1
  target_probas_2 = probas[1, [0, 1, 2], targets[1]]
  print("Batch 2:", target_probas_2)
  ```

  

- 取对数 — 为什么对数更容易优化？

```
