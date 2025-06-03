# 训练集/验证集数据比
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=False,
    shuffle=False
)
```



### 模型训练的基本流程







### 高阶的训练技巧



- 混合精度训练

混合精度是指， 使用`float32`来存储权和更新参数，在向前传播，反向传播的时候，都使用`float16`或者`bfloat16`.

在计算完梯度之后，讲更新的梯度转化回`float32`, 再进行权重参数更新。

因为`float16`表示的范围有限，当小于某个值的时，会直接变成0，这个现象叫做「下溢」。解决的方法，可以使用`bfloat16`或者损失缩放。`bfloat16`所能表示的范围和`float32`差不多，但精度比`float16`要差 。

损失缩放，讲损失函数放大，反向传播计算梯度，梯度也放大了，避免了「下溢」。 权重更新的时候，在按比例来缩小。





### 解码策略



大模型中的温度采样和top-k是什么？

温度采样是一种用于大模型解码阶段概率化的技术，能够提高生成token的多样性。

大模型是通过计算词表中，每一个词的概率来预测下一个生成的词。

词表中的每个词，都会对应一个概率值。

每次选取概率值最高的词，被称为贪婪解码。贪婪解码输出的词比较单一。

温度采样，则是按照概率值的大小，从中进行随机化抽取，选取出下一个 词。



假设词表的大小为3，模型输出单词的概率为：0.6,0.2,0.2。

对于相同的输入:

如果时贪婪解码，每次都会选取概率值最大0.6的词输出。

如果是概率化的方式，抽中0.6的概率最高，但也会随机抽到0.2的。



温度采样，是对概率的分布缩放处理。

还是以（0.6,0.2,0.2）为例。

每个概率都会除以温度采样。

当temp = 1，概率分布不变。

当temp < 1,  概率分布会锐化。大的概率值会更加凸显。

当temp > 1, 概率分布会扁平化。各个概念之间的差距会缩小。

通过设置温度采样值，来调整模型输出的丰富度。当模型输出太呆板，可以将温度采样调大，就会生成更加丰富的词。



top-k则决定哪些词会进行随机抽取的过程。

比如top-k = 30，那么只会从概率最大的前30个词中，随机抽取。

top-k能够剔除概率值极小的值，防止输出中出现无关的词。





### 模型训练的的进阶思考 



**元**



**反**



**空**

－　梯度裁剪是什么？

梯度裁剪是一种限制梯度爆炸的方法，一般采用L2范数来进行限制。在反向传播之后，得到梯度。

计算梯度的L2范数，如果大于某个阈值。就对梯度进行缩放，回到阈值之内。

L1和L2是什么？

两种常见的范数，用来衡量向量的大小。L1范数，曼哈顿距离，也就是所有元素的绝对值之和。L2范数，所有数平方，累积起来，再进行开根号。

- **L1正则化**：鼓励参数变成0，产生稀疏性（有利于特征选择）。
- **L2正则化**：鼓励参数变小但不为0，防止过拟合，让模型更平滑。

- **L1**就像你在城市街道里走路（只能横着竖着），走到目的地的总步数。
- **L2**就像你直接走直线到目的地的距离。



### 小结



## 六、如何使用LLamaFactory微调模型



### Lora微调 



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

你会发现了吗？ 将AB替换 △W之后,需要训练的参数就大大减少.  



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

### LLamaFactory的微调流程  





### 数据集构建



### 参数设置



### 开始训练 



### 数据微调的进阶思考 

### 小结



## 七、附录



## 张量的基本操作

深度学习中的输出参数和输出参数本质上都是张量。通过了解张量的变化，了解模型进行何种转化。

- 基本属性

```
x = torch.randn(3, 4, 5)

