# 查看存储信息
print(x.storage())

```

- 维度变换

  ```
  # 维度变换
  x = x.view(12, 5)      # 改变形状，要求连续
  x = x.reshape(12, 5)   # 改变形状，更灵活
  
  # 维度转置
  x = x.transpose(0, 1)  # 交换指定维度
  x = x.permute(2,0,1)   # 任意顺序重排维度
  
  # 增减维度
  x = x.unsqueeze(0)     # 增加维度
  x = x.squeeze()        # 移除大小为1的维度
  ```



- 数据转化

```
