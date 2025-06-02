# 实验一下
torch.manual_seed(123)

block_size = batch.shape[1]
ca = CausalAttention(d_in, d_out, block_size, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```



### 多头注意力

- 直接拼接输出 

```
