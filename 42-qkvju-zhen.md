# QKV矩阵



```python
# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)

# 创建查询权重矩阵，形状为 (d_in, d_out)，并且设置 requires_grad=False 表示这些权重在训练过程中不会更新
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 创建键权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 创建值权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```



```
# 使用键权重矩阵 W_key 将输入序列 inputs 投影到键空间
keys = inputs @ W_key

# 使用值权重矩阵 W_value 将输入序列 inputs 投影到值空间
values = inputs @ W_value

# 打印键向量的形状
print("keys.shape:", keys.shape)

# 打印值向量的形状
print("values.shape:", values.shape)
```



实现一个紧凑的selfAttention类 

```python
import torch.nn as nn

# 定义自注意力模块的第二个版本
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        # 调用父类构造函数
        super().__init__()
        # 设置输出维度
        self.d_out = d_out
        # 初始化查询、键和值的线性层，可以选择是否包含偏置项
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 使用线性层将输入 x 投影到查询、键和值空间
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 计算注意力分数（未归一化）
        attn_scores = queries @ keys.T
        
        # 使用 softmax 函数和缩放因子归一化注意力分数
        # 注意这里的 dim=1，表示沿着键向量的维度进行归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        # 使用归一化的注意力权重和值向量计算上下文向量
        context_vec = attn_weights @ values
        return context_vec

# 设置随机种子以确保结果的可重复性
torch.manual_seed(789)
# 创建 SelfAttention_v2 实例
sa_v2 = SelfAttention_v2(d_in, d_out)
# 使用输入数据 inputs 进行前向传播，并打印结果
print(sa_v2(inputs))
```



