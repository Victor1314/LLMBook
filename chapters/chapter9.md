# 打印注意力分数矩阵
print(attn_scores)
```

#### 权重

softmaxv归一化之后得到权重 

```
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)
```



#### 上下文向量

```
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
```

#### QKV矩阵



```python
