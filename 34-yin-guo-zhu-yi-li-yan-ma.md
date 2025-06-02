# 因果注意力掩码



因果注意力机制是一种在自注意力机制的基础之上，增加因果注意力掩码和 `dropout`机制的注意力机制。



因果注意力掩码一种剔除序列中往后的token对当前token预测影响的技术, 它能够让下一个token的预测仅基之前的token。 



如何实现因果注意力？ 



在传统的自注意力机制中， 计算权重时，会得到一个n *n的权重W矩阵。

第i行表示，第i个token对于需了中的各个token的权重。 w（i,j）表示第i个token,对于第j个token的权重分数。

对权重矩阵进行对角化处理，将对角线上方的位置对置为0。 如此一来，每个token只能看到自己之前的token。

对角线上方置为0之后，每一行的概率和不再为0，需重新进行 `softmax` 归一化处理。 



代码实现：

创建一个掩码 ：

```
# 我们创建的掩码形状应该和注意力权重矩阵的形状一致，以一一对应
block_size = attn_scores.shape[0]
# tril 方法会创建一个下三角矩阵
mask_simple = torch.tril(torch.ones(block_size, block_size))
print(mask_simple)
```





```
masked_simple = attn_weights*mask_simple
print(masked_simple)
```



再次进行归一化 

```
# dim = 1 表示按行求和
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```



另外一个思路是在softmax之前，用负无穷掩盖对角线以上的部分。

 

```
mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```



softmax之后，负无穷的部分，都是0 

```
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)
```



