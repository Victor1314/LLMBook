# 权重

softmaxv归一化之后得到权重 

```
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)
```



