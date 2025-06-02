# 计算所有标记的预测概率的对数值
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
```

- 计算平均数 

```
