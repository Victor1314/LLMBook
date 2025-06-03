# 对所有标记的概率对数值求均值
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
```

- 取负数 — 深度学习之中经常使用的是减少到0，而不是增加到0

```
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
```



使用pytorch中的entropy_loss函数，可以进行计算。

- 先在batch维度上展平这些向量 

  ```
  logits_flat = logits.flatten(0, 1)
  targets_flat = targets.flatten()
  
  print("Flattened logits:", logits_flat.shape)
  print("Flattened targets:", targets_flat.shape) 
  ```

  

- 使用交叉熵函数

```
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
```

#### 困惑度

困惑度是什么？ 

困惑度是对交叉熵进行指数计算的结果。 困惑度更有解释性，意味着模型在下一步中所不确定的词表的大小。

比如，当困惑度为10，那么意味着下一个词不确定是10中的哪一个。



#### 计算训练集和验证集的损失



```
from previous_chapters import create_dataloader_v1

