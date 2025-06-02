# 张量的基本操作

深度学习中的输出参数和输出参数本质上都是张量。通过了解张量的变化，了解模型进行何种转化。

- 基本属性

```
x = torch.randn(3, 4, 5)

# 形状
print(x.shape)  # torch.Size([3, 4, 5])
print(x.size())  # 同上

# 维度数量
print(x.dim())  # 3

# 数据类型
print(x.dtype)  # torch.float32

# 设备位置
print(x.device)  # cpu 或 cuda:0

# 总元素数量
print(x.numel())  # 3 * 4 * 5 = 60
```

- 内存相关

```
# 是否连续存储
print(x.is_contiguous())  

# 是否需要梯度
print(x.requires_grad)  

# 获取梯度
print(x.grad)  

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
# 设备转换
x = x.to('cuda')       # 转到GPU
x = x.cpu()           # 转到CPU

# 类型转换
x = x.float()         # 转为float
x = x.long()          # 转为long
x = x.bool()          # 转为boolean

# 转numpy
numpy_array = x.numpy()
# numpy转tensor
tensor = torch.from_numpy(numpy_array)
```



- 常用信息获取

```
# 最大最小值
print(x.max())
print(x.min())

# 均值标准差
print(x.mean())
print(x.std())

# 索引相关
print(x.argmax())     # 最大值索引
print(x.argmin())     # 最小值索引
```

