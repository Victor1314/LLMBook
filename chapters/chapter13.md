# 创建值权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```



```
