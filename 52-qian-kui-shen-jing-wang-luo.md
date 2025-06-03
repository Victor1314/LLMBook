# 前馈神经网络

- 线性层
- 激活函数GELU

前馈神经网络是什么?



神经网络一种模仿生物神经系统的模型。  生物大脑，接受外界的输入，经过神经元的处理，输出电信号，再传递给下一个神经元。经过数千万个神经元的协调，最终输出。



神经元是神经网络的基本单元。作为数学模型，神经元由权重、偏置、激活函数构成。

经由权重、偏置，对输入进行线性变换。

经由激活函数进行非线性变化后输出。  



最基础的神经网络由线性层和激活函数构成 。 线性层是只能进行线性变换的组件。激活函数对线性层的输出进行非线性的变换。

如下定义了一个线性层，din代表输入参数维度，dout代表输出的维度。

```　python
layer = torch.nn.Linear(100,200)  #din,dout
```



举个例子：

一个由两个线性层和激活函数构成的前馈神经网络神经网络

```python
class Feedforward(nn.Module):
   
   def  __init__(self,cfg):
       super().__init__()
       self.layers = nn.Sequential(
       		nn.Linear(cfg["emb_dim"], 4* cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4* cfg[ "emb_dim"], cfg["emb_dim"])
       )
    
    def forward(self,x):
        return self.layers(x)
    

```

思考：假设 cfg["emb_dim"] = 256, 上面这个前馈神经网络有多少个参数？参数中的权重矩阵参数和偏置项分别是多少？

提醒：偏置项个数等于输出维度。

让我们来实现GPT中的前馈神经网络　



GELU函数

```
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```



完整的前馈神经网络

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)
```



