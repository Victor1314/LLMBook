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
