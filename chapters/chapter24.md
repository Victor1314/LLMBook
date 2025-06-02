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



#### dropout机制

 使用`dropout`  ，会随机按照一定的比例，将权重矩阵种的元素置为0.



```
