# 交叉熵



如何计算交叉熵？  

- 目标词元的概率

  - logits
  - target-token

  ```
  inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                         [40,    1107, 588]])   #  "I really like"]
  
  targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                          [588,  428,  11311]]) #  " really like chocolate"]
  ```

  

  ```
  with torch.no_grad():
      logits = model(inputs)
  
  probas = torch.softmax(logits, dim=-1) # 词表中每个标记的预测概率
  print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
  ```

  

  ```
  batch_idx = 0
  target_probas_1 = probas[batch_idx, [0, 1, 2], targets[batch_idx]]
  print("Batch 1:", target_probas_1)
  
  batch_idx = 1
  target_probas_2 = probas[1, [0, 1, 2], targets[1]]
  print("Batch 2:", target_probas_2)
  ```

  

- 取对数 — 为什么对数更容易优化？

```
# 计算所有标记的预测概率的对数值
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
```

- 计算平均数 

```
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

