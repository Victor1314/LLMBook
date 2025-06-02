# 层归一化



层归一化是一种增加模型训练稳定性的深度学习技术， 能避训练过程中的梯度消失或者梯度爆炸。

层归一化的流程为，调整神经网络层的输出，让它符合：”均值为0，方差为1“的规则。

再将层归一化之后的输出作为下一个神经网络层的输入。

具体步骤为：

- 求输出的均值和方差
- 输出中各个维度的值减去均值 ； 除以方差的平方根

在transformer block中，两次运用了层归一化。分别在因果注意力机制输入之前，前馈神经网络输入之前。 



下面层归一化层的实现

```
 class Normlayer(nn.Module):
 	def __init__(self,emb_dim):
 		super().__init__()
 		self.eps = 1e-5
 		self.scale = torch.nn.Parameter(torch.ones(emb_dim))
 		self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
 	def forward(self,x):
		mean = x.mean(-1,keepdim= True)
		var  = x.var(-1,keepdim = True,unbiased=False)
		x = (x -mean)/torch.sqrt(var+ self.eps)
		return  self.scale *x + self.shift
 	
```

unbiased=False 的意思是使用有偏方差，原因在于数据量比较大，误差可以忽略不记。

