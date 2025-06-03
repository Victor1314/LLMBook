# 权重分数

```
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
```



```
# 创建一个 6x6 的零张量，用于存储注意力分数
attn_scores = torch.empty(6, 6)

# 遍历输入序列中的每个元素
for i, x_i in enumerate(inputs):
    # 对于当前的输入元素 x_i，再次遍历整个输入序列
    for j, x_j in enumerate(inputs):
        # 计算 x_i 和 x_j 的点积，作为注意力分数，并存储在 attn_scores 矩阵的对应位置
        attn_scores[i, j] = torch.dot(x_i, x_j)

# 打印完整的注意力分数矩阵
print(attn_scores)
```



或者可以通过矩阵乘法计算

```
# 使用矩阵乘法计算输入序列的点积矩阵
# inputs @ inputs.T 相当于 inputs 与 inputs 的转置相乘
attn_scores = inputs @ inputs.T

# 打印注意力分数矩阵
print(attn_scores)
```

