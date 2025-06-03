# Lora微调



Lora微调是什么？ 



Lora微调(Low-Rank Adaptation)，是一种高效率的微调技术。可以降低微调参数的数量，从而降低微调所需要的资源。



Lora的论文地址：https://arxiv.org/abs/2106.09685



Lora微调为什么高效？ 

让我们来看下Lora的原理。

对于全参微调的过程，可以简单表示

W1= W  + △W

 W是微调前的权重参数，△W是权重的更新，W  + △W得到新的权重参数W１。需要注意，△W在微调的过程中更新，和W具有一样的维度大小。

Low-Rank Adaptation，Rank是矩阵的秩，秩是矩阵中线性无关的向量值，不会超过最小的维度值。比如一个 30* 2的矩阵，rank 最大为2.

降低秩，某种程度上可以减少训练权重。

Lora做了一个这样的事情。将 △W 近似成 AB, 两个矩阵相乘.

W1≈  W   +  AB

A和B是一个秩相对很小的矩阵, 一般设置为16.

发现了吗？ 将AB替换 △W之后,需要训练的参数就大大减少.  



从工程实践的角度来看,  Lora参数可以单独训练, 单独保存, 推理的时候再和原始的权重一起加载. 

这个得益于矩阵乘法乘法的分配律,   

XW1  = XW +  XAB

X为模型的输入，求解输出的时候，我们可以分开计算原模型的输出XW 和Lora模型的输出XAB，再把他们拼接起来。



如何在码层面实现Lora？ 

首先，我们要定义一个类LoraLayer,  继承 Module。 传入参数包括d_in, d_out, rank，以及以及缩放参数alpha。 

包含两个权重矩阵:A和B, A初始化，B全为0.

forwad函数定义为 缩放alpha*  （(X @B @C)。

```py
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

```

`代码中使用了torch.nn.init.kaiming_uniform_` ,是一中对权重参数初始化的方法，它能确保A在开始训练时，不会突然梯度消失或者爆炸，有良好的收敛效果。



在有了基本的Lora类之后，我们需要实现上面提到的拼接输出的功能。也就是把常规的线性层和Lora类组合起来形成新的类，forwad函数为两个的拼接输出。

```py
class LinearWithLoRA(torch.nn.Module):
	def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
        linear.in_features,
        linear.out_features,
        rank,
        alpha
)
    def forward(self, x):
    	return self.linear(x) + self.lora(x)
```

再就是需要实现一个函数，把模型中的线性层，都替换成待有Lora的线性层。

```py
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 如果是线性层，替换为 LoRA 版本
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 如果不是线性层，递归处理其子模块
            replace_linear_with_lora(module, rank, alpha)
```

其中 `model.named_children()` 是 `nn.Module` 中返回子模块的方法，返回对象是一个元组。仅包含子模块，不包含子模块的子模块，所以下面要采用递归调用的方式。

id: 20250511210355
ref:《从零构建大模型》
changeLog: 

- 20250511 Init



