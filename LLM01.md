# 零基础入门大模型



## 前言



本书的设计有以下**五大特色**：

- 从感性认识出发 — 无需任何大模型知识，手把手带你手搓一个自己的模型

- 从最经典的GPT模型出发 — 带你深入GPT2的各种细节，掌握大模型的核心原理
- 从最先进的模型出发 —　深入qwen2.5的参数，让你可以基于各类场景，进行微调
- 从最实用的工具出发　— 　使用目前业内广泛使用的微调工具 Llama-factory , 让你快速微调各类模型
- 从高阶出发　—　利用元反空升阶。比如，GPT2使用的是传统的MHA，在进阶部分我会总结MHA的不足，还有哪些模型，让你无缝衔接到Qwen2.5使用的GQA。





本书有两个线索，一条是主线，是带领深入最经典的GPT2模型，带你手把手训练一个GPT２。　一条是暗线，通过基础原理的讲解，以及高阶思考中的元反空升阶，支撑你去探索qwen2.5以及别的模型，理解它们的创新之处。 在这个旅途之中，你会掌握大模型最核心的原理，一套微调的工具和方法论。





### 如何使用本书

### 如何训练一个大模型？ 

#### pretrain

#### rlhf

#### sft

### qwen2.5





#### 模型架构

#### 基本参数

#### 使用vllm部署qwen2.5



### pytorch常用语法

 

#### 神经网络的基本操作





#### 张量的基本操作

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



## 一、快速开始 —  训练你的第一个模型

采用预训练(pretrain)和监督微调（Supervised Fine-Tuning，sft）的方法，以最小的成本复现 `minimind-zero`, 加速通关从模型预训练，微调，到部署。获取感性
<!--more-->

---

### 💻 环境准备


- 以下我的软硬件配置：
  - `windows10` 
  - `anaconda`
  - `python10.6`
  - `GPU`云服务平台：[modal](https://modal.com/)

1. 克隆项目

```
git clone https://github.com/jingyaogong/minimind.git
```

  安装依赖, 进入`minimind`项目。 使用 `anaconda`或者 `uv`的同学，可以先创建虚拟环境，再安装。

```
(minimind) D:\minimind>pip install -r requirements.txt
```

2. 配置 `modal` 

 `modal`  是一个GPU云服务平台。如果你本地没有 `GPU`, 可以使用云服务平台来训练模型。 

新手入门，四步走如下：

- 首先，在官网注册。官网地址： https://modal.com/

- 本地安装 `modal` 的客户端，配置 `key` 。有`python`环境和`pip`包的话，以下两个命令搞定

```
pip install moda
python -m modal setup
```

注意：需要提前安装 `pip`  和 `python` 。 

- 编写脚本文件 。在脚本文件中编写执行训练功能的函数，配置训练时的相关参数，比如：`GPU`调用，文件存储等。
- 使用 `modal` 命令，执行脚本中相关的函数。

![img](https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/59476626/kEWGm3ZmXKHKnvcY.png?Expires=1743685818&Signature=OMHlacDQErvSa7wnB4ifIorqCDpmt4DtA34Qce0hcM111ugBJ~dwSFdurk61SQpC7cwEQ~uQUyMOScEkivoz1Cvz6VynJxUu~hbBATDeOpdfKQSWg4gPbBLSORmT3I2qk5n8hMxEEGpGqRm5ttYvIeKGj2cH5o6zPH0-R2PeZs9~KlfwiuKhBE7rfRLCAPfXTD6mxpsMyz2BagA34G1Bp~3TAqp0M8fV0ZJGLo5BM98hak7t215-wjCP22Rb9kqeJ8P780b9Zk8kcnZ7OK367Vv46DO14N5SYug1biXeGxLPw3p76Sd0NoBAZ~kvn~lcnMyKndu-l1pQ2dGcK9ctWw__&Key-Pair-Id=K3USGZIKWMDCSX)

**tips**:  注册 `modal` 赠送5$，不太够用。幸运的是绑定信用卡可以每月赠送30$. 我用的是虚拟信用卡，参考:[nobepay](https://www.nobepay.com/)  或者 `visa`卡。



4. `modal` 脚本

 `modal`脚本命名为： `modal.train.py` , 放在 `minimind` 的根目录下面。下面为训练 `minidmind-zero` 用到的

脚本文件。 

主要进行的操作为：定义镜像，安装相关的依赖, 导入数据集文件；创建 `volume` , 存储训练后的模型文件;定义 `pretrain`  `sft` 的训练函数，包括：执行 `minimind`中的脚本，定义执行相关的参数。

```
import modal
from datetime import datetime  
app = modal.App("minimind-training")

# 定义镜像
training_image = (modal.Image.debian_slim()
    .pip_install([
     
    ])
    .apt_install(
    "git"  # 添加 git 安装
       )
    .run_commands([
        "git clone https://github.com/jingyaogong/minimind /root/minimind",
        "cd /root/minimind && pip install -r requirements.txt",
         "mkdir -p /root/minimind/out"  # 确保输出目录存在
    ])
    .add_local_dir("dataset", remote_path="/root/minimind/dataset")  # 使用add_local_dir
)


# 使用volume确保数据持久化
volume  = modal.Volume.from_name("my-volume")  # my-volume 替换为你创建的volume名称

@app.function(
    image=training_image,
    gpu="A100",  # 请求使用 GPU
    timeout=3600,  # 设置超时时间为1小时
    max_containers=1,  
    volumes={"/root/minimind/out": volume}
)
def train_pretrain():
    import os
    os.chdir("/root/minimind")
    os.system("python train_pretrain.py")


@app.function(
    image=training_image,
    gpu="A100",  # 请求使用 GPU
    timeout=3600,  # 设置超时时间为1小时
    max_containers=1,  
    volumes={"/root/minimind/out": volume}
)
def train_sft():

    import os    
    volume.reload()
    os.chdir("/root/minimind")
    os.system("python train_full_sft.py")

```

对于训练函数的设置，可以根据训练函数，调整 `gpu` 类型和 `timeout`。注意，尽可能设置 `timeout` 长一些，否则会容易超时。

```
@app.function(
    image=training_image,
    gpu="A100",  # 请求使用 GPU
    timeout=3600,  # 设置超时时间为1小时
    max_containers=1,  
    volumes={"/root/minimind/out": volume}
)
```



`my-volume` 是你创建的 `volume`名称，从https://modal.com/storage中查看，需要提前创建，命令如下。

```
% modal volume create my-volume
Created volume 'my-volume' in environment 'main'.
```



5. 数据集

需要用到 `pretrian` 和 `sft`  各自的数据集.  下载地址: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files 

仅需要下载 `pretrain_hq.jsonl` 和 `sft_mini_512.jsonl`，放到项目 `miminid/dataset` 下。

在脚本中，我们将 `dataset` 挂载进 容器的 `/root/minimind/dataset` 目录。

```
.add_local_dir("dataset", remote_path="/root/minimind/dataset")  # 使用add_local_dir
```



至此，相关环境已经准备好了。下面即将进入激动人心的训练环节。在此次请检查：

- 成功安装 `modal` 客户端
- `modal` 上创建了 `volume`，`modal.train.py`中的`volume`名称要替换成你所创建的 `volume` 名称
- `minimind`根目录下存在 `modal.train.py` 脚本
- `minimind/dataset` 内存在`pretrain_hq.jsonl` 和 `sft_mini_512.jsonl`这两个数据集。

---


### 🚀 模型训练


通过**终端**进入到 `minimind`目录

#### 预训练(pretrain) —— 让模型学习海量知识

执行预训练的脚本.

```
 modal run modal_train.py::train_pretrain
```

![image-20250320232517933](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250320232517933.png)



从日志输出,看出多少个Epoch，执行了多少步，loss值和学习率为多少。

![image-20250321000118852](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250321000118852.png)

通过生成的链接，可以远程访问训练情况

![image-20250320232658106](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250320232658106.png)

相关的指标,可以查看训练时的系统的运行情况。

![image-20250321001112385](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250321001112385.png)



训练结束，可以从从 `volume` 查看训练结果。

![image-20250321165059127](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250321165059127.png)



#### 监督微调(sft) — 让模型学会说话

执行命令

```
 modal run modal_train.py::train_sft
```

![image-20250321171223179](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250321171223179.png)

结束之后，出现结果文件。

![image-20250322085053992](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322085053992.png)



下载`volume`中的训练结果到本地 `\minimind\out`文件夹中。

```
modal volume get my-volume full_sft_512.pth pretrain_512.pth
```

![image-20250322090720246](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322090720246.png)

至此，训练结束，让我们来测试一下。

`minimind`项目中提供运行模型的脚本, 直接执行就可以。执行之前可以看下相关参数，比如：`model_mode` 指的是执行什么模型

```
 parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
```


测试 `sft` 模型

```
python eval_model.py --model_mode 1
```



选择手动测试

![image-20250322094629054](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322094629054.png)

 选择自动测试 

![image-20250322094857989](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322094857989.png)

###  🌐 模型部署

通过将训练好的模型上传 `huggingface`  ,并且通过`huggingface` 的 `space` 创建运行模型的`demo` 

https://huggingface.co/docs 

####  转化为HF格式

目前是的 `pytorch` 的格式,  需要先转化成 `HF`格式，才能上传到 `huggingface`。模型文件在`minimind\out\full_sft_512.pth` 。

格式问题，可以参考下面这篇文章。https://mp.weixin.qq.com/s/HXMLPM2KNkO6Mah-4btaEQ

项目中提供了转化的脚本, 在 `srcipts\convert_model.py` 。

源代码

```
if __name__ == '__main__':
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=8192, use_moe=False)

    torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"

    transformers_path = '../MiniMind2-Small'

    # convert torch to transformers model
    convert_torch2transformers(torch_path, transformers_path)
```

这里修改将 `stf` 之后的模型进行转化.**请替换模型路径，修改你的模型名称**

```
if __name__ == '__main__':
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=8192, use_moe=False)

   # torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"

    transformers_path = '../MiniMind-zero'  # 你想要的名称
    torch_path = "../out/full_sft_512.pth"  # 你的.pth文件路径
    # convert torch to transformers model
    convert_torch2transformers(torch_path, transformers_path)

    # # convert transformers to torch model
    # convert_transformers2torch(transformers_path, torch_path)
```

执行脚本之后, 在根目录下出现`MimiMind-zero`文件夹

![image-20250322201705561](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322201705561.png)

![image-20250327222336660](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327222336660.png)



####  上传到 Huggingface 

新建模型：https://huggingface.co/new-space

![image-20250322204200340](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250322204200340.png)

通过网页直接上传模型。（中间折腾过git, huggingface-cli，发现直接在**网页上传坑是最少**的）

![image-20250323074542851](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250323074542851.png)

修改 `readme` 文件，添加相应的标签

```
language: zh
tags:
  - pytorch
  - gpt
  - transformers
  - text-generation-inference
library_name: transformers
pipeline_tag: text-generation
inference: true
```

至此你的模型已经上传好了。

https://huggingface.co/cmz1024/minimind-zero/tree/main

![image-20250327222957713](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327222957713.png)

####   创建Space，show出你的模型

创建space,https://huggingface.co/new-space.我选择的是通过 `Gradio` 创建，并勾选`chatbot`。

![image-20250327224010360](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327224010360.png)



创建成功会进来这里

![image-20250327224103524](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327224103524.png)

![image-20250327224139074](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327224139074.png)

进入的 `app.py` ，发现是通过 `InferenceClient` 来调用模型。这是 `HF` 提供的 `Inference API` 。但是咱们的模型是基于 `pytorch`构建，要使用`Inference API` 必须要用 `transformers` 来构建。

```
import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta") # hugginface 提供的Inference API
```

当然也可以转化，为了方便，我采用直接在 `Gradio` 加载 `pytorch` 模型。

大家可以到这里[下载代码](https://cnb.cool/huoshuiai/LLM101-chenmuzhi/-/blob/main/app.py)，替换`app.py`， 其中要修改的是你的模型路径。

```
# 加载模型和分词器
model_path = "cmz1024/minimind-zero"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
```



至此，你就创建了一个展示demo, 把你的喜悦和进步分享给你的朋友吧。

![image-20250327201730558](https://for-note.oss-cn-shanghai.aliyuncs.com/img/image-20250327201730558.png)







### 快速开始的进阶思考 





### 小结

## 二、数据集



### 如何对文本进行编码？Tokenizer 



数据处理的第一步，就是使用数字对文本进行编码。 

Tokenizer，分词器，将文本划分成一个个最小单元 —词元(token),  用于模型训练。

以BERT、GPT等预训练语言模型为例，tokenizer的作用流程如下：

1. 输入文本："I love AI."
2. 分词器处理后：["I", "love", "AI", "."]
3. 再转成ID：[101, 2001, 727, 102]（假设的ID）
4. 输入到模型中。

词表(Vocabulary)是分词器的重要组件，将token转化成token_id。在训练前的语料处理，模型输出的解码阶段，都会用到词表。 词表重要参数就是词表大小，词表越大，能生成的语言越多。

常见的分词器有BPE、wordPiece等。 BPE在大模型广泛使用，而wordPiece则是在Bert中使用。两者都属于

子词分词器。



**什么是子词？**

子词是介于词和字符之间单元。 假如使用词作为单位来划分文本的话，颗粒度太大，词表无法兼容所有的词汇。 容易出现陌生词，也就是OOV问题(out-of-vocabulary) 。 假如使用字符来进行分词，颗粒度太小，token没有实际的含义。

子词兼有两者的优点。

比如： 常见英文单词的前缀和后缀，re，ful，ly。



**BPE(byte pair encoding)算法的原理 .**

它是如何训练，构建词表的呢？

先将要训练的文本集合，按照字符来进行拆分。拆分之后，将相邻字符组合，构成子词，统计子词的频率。

假设文本为，“Ｉ　love you”, 那么统计 I , Io,ov, yo, ou 的频率。

将频率高的子词，确定进行合并。 

再进行下一轮统计。将子词和相邻的字符或者子词合并，合并频率最高的。

直到达到词表的大小。



**wordPiece分词的原理。**

wordPiece是子词分词算法，在Bert等语言模型之中广泛使用。wordPiece和BPE的区别在与在进行子词合并的时候，考虑了语料的概率。也就是说，在分词之后，要确常见子词的概率。而BPE中仅根据出现频率来进行考虑。出现频率高的，不一定是常见子词。

假设我们有如下简单语料：

```
复制unhappiness
unhappy
unhappily
unhappiest
```

**1. BPE 的处理方式**

BPE 会统计所有连续字符对的出现频率，比如：

- “un”、“ha”、“pp”、“in”、“es”、“ly”、“est”等等

假设“pp”在语料中出现频率很高（比如在“happiness”、“happy”、“happily”、“happiest”里都出现了），BPE 可能会优先把“pp”合并成一个子词。

但有时候，某些字符对虽然频率高，却并不是有实际意义的子词（比如“pp”本身在英文中没有独立意义）。

**2. WordPiece 的处理方式**

WordPiece 不仅考虑“pp”的出现频率，还会计算如果把“pp”合并成一个子词，是否能显著提升整个语料的概率（即更好地表示原始语料中的单词）。

假设“un”、“happy”、“ness”、“ly”、“est”这些子词在英语中很常见，WordPiece 可能会更倾向于合并这些有实际意义的子词，而不是仅仅频率高但没意义的“pp”。

比如，WordPiece 可能会优先得到如下子词：

- “un”
- “happy”
- “ness”
- “ly”
- “est”

这样，“unhappiness”会被分成“un + happy + ness”，而不是“un + ha + pp + in + ess”。

| 单词        | BPE 分词结果            | WordPiece 分词结果 |
| ----------- | ----------------------- | ------------------ |
| unhappiness | un + ha + pp + in + ess | un + happy + ness  |
| unhappily   | un + ha + pp + ily      | un + happy + ly    |
| unhappy     | un + ha + pp + y        | un + happy         |



### embedding — 把数字变成向量

经过分词之后，我们得到文本对应的token_id，也就是一串数字。



为什么需要把token_id转化成向量？

token_id 只是一个整数编号，本身没有任何语义信息。神经网络擅长处理**向量**（一组有意义的数字），而不是单一的整数。也就是说，只有转化成向量，才能进行语义表示，才能进行训练。



如何转化？

代码xx 



### 如何对位置进行编码？ 

位置编码是一种将



### 构建Input- target训练对



### 常见的数据集格式

#### AIpaca



#### ShareGPT 





### 数据集的进阶思考 

**元**

**反**

**空**

### 小结

## 三、自注意力机制 



### 为何Transfomer架构能够胜出？



先看看transformer之前的RNN和CNN. 



RNN和LSTM是什么 ? 

1. RNN（循环神经网络，Recurrent Neural Network）

**基本原理**

- RNN是一类用于处理序列数据的神经网络。
- 不同于传统的前馈神经网络，RNN在每个时间步都会接收当前输入和上一个时间步的“隐藏状态”作为输入，实现信息的“记忆”与传递。

**优点**

- 能处理变长的序列数据（如文本、语音、时间序列等）。
- 结构简单，参数共享，适合序列建模。

**缺点**

- **梯度消失/爆炸**：长序列训练时，梯度会迅速变小或变大，导致模型难以捕捉长距离依赖关系。
- 训练效率较低，不能并行。

LSTM（长短期记忆网络，Long Short-Term Memory）

**基本原理**

- LSTM是RNN的改进版，专门为了解决RNN的“长距离依赖”问题（即梯度消失/爆炸）。
- 通过引入“门控机制”，控制信息的“记忆”与“遗忘”。LSTM单元包含三个门：输入门、遗忘门、输出门。

| 特点       | RNN               | LSTM                     |
| ---------- | ----------------- | ------------------------ |
| 结构       | 简单              | 复杂（有门控）           |
| 长距离依赖 | 容易丢失          | 能较好捕捉               |
| 参数量     | 少                | 多                       |
| 训练难度   | 梯度消失/爆炸严重 | 缓解梯度消失/爆炸        |
| 计算速度   | 慢（不能并行）    | 慢（不能并行）           |
| 应用       | 简单序列建模      | 复杂序列建模，长依赖场景 |



transformer的优势在哪里？  

1. 并行计算能力强
RNN/LSTM：序列数据必须按时间步依次处理，不能并行（即第t步的输出依赖于第t-1步的输出），导致训练和推理速度慢。
Transformer：基于自注意力机制，所有位置的输入可以同时处理，实现完全并行，大幅提升训练效率。
2. 捕捉长距离依赖能力强
RNN/LSTM：虽然LSTM通过门控机制缓解了梯度消失问题，但长距离依赖仍然难以捕捉，信息传递路径长，容易丢失上下文信息。
Transformer：自注意力机制可以直接建立任意两个位置之间的联系，无论距离多远，捕捉长距离依赖效果更好。
3. 建模灵活
RNN/LSTM：只能顺序建模，难以处理非顺序结构的数据。
Transformer：通过自注意力机制，可以灵活地建模序列中任意位置之间的关系，更适合复杂结构的数据。
4. 扩展性强
RNN/LSTM：堆叠层数受限，层数多了易出现梯度消失或爆炸。
Transformer：结构简单，易于堆叠更深的网络层，提升模型容量和表达能力。
5. 更适合大规模数据和预训练
Transformer结构非常适合大规模数据的分布式训练，也是BERT、GPT等预训练模型的基础结构。
总结
RNN/LSTM：顺序处理、依赖前后关系、捕捉长距离依赖弱、不易并行。
Transformer：全局自注意力、并行处理、捕捉长距离依赖强、易扩展和预训练。



### 自注意力机制

自注意力机制是什么？  



自注意力机制是一种对**同一序列**内不同token之间的关系编码的技术，也就是能够将token和token之间的关系进行编码。

通过自注意力机制，序列中的token都会得到一个上下文向量（context -vector), 用于表示序列中上下文的信息。



**简单的自注意力机制**不包含**权重矩阵**，仅用于演示注意力机制的简单流程。权重矩阵是模型中可训练的参数。 



简单的自注意力机制的**实现流程**：

遍历序列中的token所对应的向量值。

计算当前向量和其它向量的点积，得到权重分数，权重分数经过 `softmax`函数归一化得到权重值。

序列中的所有向量 * 权重数进行累加，就得到当前token对应的上下文向量。



举例：

序列为：”我爱阅读“

对应的向量：(vector1, vetor2,vector3,vector4 )

当前的token为”我“，那么会用 vector1 *  vectori （i = 1,2,3,4）得到权重分数(score1,score2,score3,score4)

权重分数归一化之后得到(w1,w2,w3,w4)

最终得到，”我“对应的 context_vector1 =   w1 *vecotr1 + w2 * vector2 + w3 * vector3 + w4* vector4。 



在第二章根据token创建embeding的时候，仅考虑了单个toekn以及token在序列中的位置，没有考虑token之间的关系。

自注意力机制弥补了这个缺点，经过自注意力机制得到的上下文向量，能更好的表示token的，让模型获得更好地输出。





#### 权重分数

```
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
```



```
# 创建一个 6x6 的零张量，用于存储注意力分数
attn_scores = torch.empty(6, 6)

# 遍历输入序列中的每个元素
for i, x_i in enumerate(inputs):
    # 对于当前的输入元素 x_i，再次遍历整个输入序列
    for j, x_j in enumerate(inputs):
        # 计算 x_i 和 x_j 的点积，作为注意力分数，并存储在 attn_scores 矩阵的对应位置
        attn_scores[i, j] = torch.dot(x_i, x_j)

# 打印完整的注意力分数矩阵
print(attn_scores)
```



或者可以通过矩阵乘法计算

```
# 使用矩阵乘法计算输入序列的点积矩阵
# inputs @ inputs.T 相当于 inputs 与 inputs 的转置相乘
attn_scores = inputs @ inputs.T

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
# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)

# 创建查询权重矩阵，形状为 (d_in, d_out)，并且设置 requires_grad=False 表示这些权重在训练过程中不会更新
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 创建键权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 创建值权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```



```
# 使用键权重矩阵 W_key 将输入序列 inputs 投影到键空间
keys = inputs @ W_key

# 使用值权重矩阵 W_value 将输入序列 inputs 投影到值空间
values = inputs @ W_value

# 打印键向量的形状
print("keys.shape:", keys.shape)

# 打印值向量的形状
print("values.shape:", values.shape)
```



实现一个紧凑的selfAttention类 

```python
import torch.nn as nn

# 定义自注意力模块的第二个版本
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        # 调用父类构造函数
        super().__init__()
        # 设置输出维度
        self.d_out = d_out
        # 初始化查询、键和值的线性层，可以选择是否包含偏置项
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 使用线性层将输入 x 投影到查询、键和值空间
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 计算注意力分数（未归一化）
        attn_scores = queries @ keys.T
        
        # 使用 softmax 函数和缩放因子归一化注意力分数
        # 注意这里的 dim=1，表示沿着键向量的维度进行归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        # 使用归一化的注意力权重和值向量计算上下文向量
        context_vec = attn_weights @ values
        return context_vec

# 设置随机种子以确保结果的可重复性
torch.manual_seed(789)
# 创建 SelfAttention_v2 实例
sa_v2 = SelfAttention_v2(d_in, d_out)
# 使用输入数据 inputs 进行前向传播，并打印结果
print(sa_v2(inputs))
```



### 因果注意力





#### 因果注意力掩码



因果注意力机制是一种在自注意力机制的基础之上，增加因果注意力掩码和 `dropout`机制的注意力机制。



因果注意力掩码一种剔除序列中往后的token对当前token预测影响的技术, 它能够让下一个token的预测仅基之前的token。 



如何实现因果注意力？ 



在传统的自注意力机制中， 计算权重时，会得到一个n *n的权重W矩阵。

第i行表示，第i个token对于需了中的各个token的权重。 w（i,j）表示第i个token,对于第j个token的权重分数。

对权重矩阵进行对角化处理，将对角线上方的位置对置为0。 如此一来，每个token只能看到自己之前的token。

对角线上方置为0之后，每一行的概率和不再为0，需重新进行 `softmax` 归一化处理。 



代码实现：

创建一个掩码 ：

```
# 我们创建的掩码形状应该和注意力权重矩阵的形状一致，以一一对应
block_size = attn_scores.shape[0]
# tril 方法会创建一个下三角矩阵
mask_simple = torch.tril(torch.ones(block_size, block_size))
print(mask_simple)
```





```
masked_simple = attn_weights*mask_simple
print(masked_simple)
```



再次进行归一化 

```
# dim = 1 表示按行求和
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```



另外一个思路是在softmax之前，用负无穷掩盖对角线以上的部分。

 

```
mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```



softmax之后，负无穷的部分，都是0 

```
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)
```



#### dropout机制

 使用`dropout`  ，会随机按照一定的比例，将权重矩阵种的元素置为0.



```
# 随便设置一个随机数种子
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # 设置 50% 的 Dropout 比例
# 对注意力权重进行 dropout
print(dropout(attn_weights))
```



实现一个因果注意力类 



```
# 定义一个带 dropout 的因果自注意力层
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        '''
        构造函数，输入参数如下：
        d_in: 输入的维度
        d_out: 输出的维度
        block_size: 注意力权重矩阵的大小
        dropout: dropout 比例
        qkv_bias: 是否对 query、key 和 value 加偏置
        '''
        super().__init__()
        self.d_out = d_out
        # 根据前文，每一个权重矩阵都是 d_in x d_out 的线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 一个 dropout 层
        self.dropout = nn.Dropout(dropout) 
        # 一个掩码矩阵，下三角为 1，其余为 0
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1)) # New

    def forward(self, x):
        '''
        前向传播函数，输入参数为 x，维度为 b x num_tokens x d_in，输出维度为 b x num_tokens x d_out
        '''
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # transpose 是为了实现矩阵乘法
        attn_scores = queries @ keys.transpose(1, 2)
        # 即上文说过的，将掩码从 0 修改为 -inf，再进行遮蔽操作
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 经过 softmax 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        # 进行 dropout
        attn_weights = self.dropout(attn_weights) # New
        # 得到最后结果
        context_vec = attn_weights @ values
        return context_vec

# 实验一下
torch.manual_seed(123)

block_size = batch.shape[1]
ca = CausalAttention(d_in, d_out, block_size, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```



### 多头注意力

- 直接拼接输出 

```
# 定义一个多头注意力层
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
            # 将 num_heads 个单头注意力层组合在一起来实现多头
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, block_size, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # 前向计算时将多个头的输出拼接在一起
        return torch.cat([head(x) for head in self.heads], dim=-1)


# 实验一下
torch.manual_seed(123)

block_size = batch.shape[1] # token 数量
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```



- 通过权重矩阵分割

另外一种方式是，通过张量重塑QKV矩阵，来进行计算。　



```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 因为要对权重矩阵按注意力头数进行拆分，所有输出维度必须是头数的整数倍
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # head_dim 就是拆分之后每个头应该输出的维度
        self.head_dim = d_out // num_heads 

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 形状为 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 我们可以通过增加一个 num_heads 的维度来将矩阵分割到每个头
        # 维度变化: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置一下: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力权重
        # 基于矩阵乘法，简单地实现各个头的并行计算
        attn_scores = queries @ keys.transpose(2, 3) 
        # 一般来说我们会将掩码矩阵转化为 bool 值并基于序列的长度进行截断
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 需要将掩码矩阵 unsqueeze 两次，也就是增加两个维度，才能让掩码矩阵的维度和注意力权重对应上
        mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
        # 使用掩码矩阵来进行遮蔽
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # 将多个头的输出重新组合回去 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

# 试验一下
torch.manual_seed(123)

batch_size, block_size, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```



### 注意力机制的进阶思考 

**元**

本质上是XXX

**反**

多头注意力机制的计算量过大。  



**空**

- MHA，ＭＱＡ，GQA对比:

MHA ，多头注意力机制，transformer论文中提出。

ＭＱＡ， MHA 改进版，每个头有一个Q，所有的头共享KV矩阵。

GQA，　综合版，每个头有一个Q，有多组的KV。

| 机制 | Query    | Key/Value  | 参数量/速度   | 表达能力 |
| ---- | -------- | ---------- | ------------- | -------- |
| MHA  | 每头独立 | 每头独立   | 参数最多/慢   | 最强     |
| MQA  | 每头独立 | 全部头共享 | 参数最少/最快 | 较弱     |
| GQA  | 每头独立 | 每组头共享 | 参数适中/较快 | 适中     |



- FlashAttention是什么？

flashAttention是一种高效的注意力机制实现方式。 传统的注意力计算的方法, 需要计算QK(T)矩阵相乘，时间复杂度为O(n2), 再对大矩阵进行softmax.  flashAttention的核心为，将大矩阵分成block，以block作为核心计算矩阵乘法运算，以及归一化计算。有了flashAttention之后，可以提高计算的效率。

```
┌──────────────┐
│   Q, K, V    │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 分块加载小块 │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 计算分数     │
│ 边softmax    │
│ 边加权求和   │
└─────┬────────┘
      │
      ▼
┌──────────────┐
│ 处理下一个块 │
└──────────────┘

```



- KVcache的原理

大模型一次吐出一个词。在原始的transformer中，在推理的时候，需要计算重新计算历史token的KV，新的token要生成KVQ，用Q去查询历史token的KV，捕捉特征。 因此历史的KV完全可以复用 。

kv-cache 的 shape 是

[batch_size, n_heads, past_seq_len, head_dim]

past_seq_len：历史token长度，会随着生成步数增长。 

kv-cache面临内存增长的问题，目前有的办法是将模型分到不同的GPU上，或者采用量化的方式减少显存使用。

| 名称       | 维度（多头）                             | 用途              |
| ---------- | ---------------------------------------- | ----------------- |
| Q          | [batch, n_heads, seq_len, head_dim]      | 当前token查询向量 |
| K          | [batch, n_heads, seq_len, head_dim]      | 历史token键向量   |
| V          | [batch, n_heads, seq_len, head_dim]      | 历史token值向量   |
| kv-cache K | [batch, n_heads, past_seq_len, head_dim] | 缓存历史K         |
| kv-cache V | [batch, n_heads, past_seq_len, head_dim] | 缓存历史V         |





- VLLM的加速原理

pageAttention 和 Continuous Batching

传统的kv-cache在进行管理的时候，会分配一个大张量，张量的size为(layer, batch_size, sequence_length, heaer_count,  head_dim)

sequence_length 通常会padding到同一个长度，导致显存浪费。并且分配大张量，会导致显存碎片。

你猜到了吗？ pageAttention 和内存的分页管理思想一样, 不再给请求分配一个大块显存。而是给先将显存分割成较小的页，同一个请求可以横跨不同的页。多个请求的kv可以拼接在一个大张量里。这样不会造成padding浪费。

当请求过来，通过**页表**查询历史token在那一页。

pageAttention,  kv进行矩阵运算的时候，是不是要copy一份? 　并不是。

 在做Attention计算时，**通过高效的索引（gather/scatter）操作，把需要的kv在计算时“逻辑上拼成”一个连续矩阵**

**Continuous Batching**  , 能够将不同时间发起来的请求，拼接到一个batch里推理,GPU利用率更高，相应更快。

也就是能动态的扩展当前batch 



### 小结





## 四、实现一个GPT模型

### transformer block



ransformer block 是大模型的核心组件。模型参数中的层数`num_layers`，指的就是transformer block的个数。gpt2的`num_layers`为48，gpt3为96.

 它包括因果注意力机制和前馈神经网络, 以及在进入因果注意力和前馈神经网络层之前的层归一化。并且通过快捷连接来连接因果注意力层和前馈神经网络层。 

当输入X进来，transformer block如何运转呢？  X会依次经过因果注意力机制和前馈神经网络层的处理。



首先，X会经过被归一化，处理成均值为0，方差为1的向量值 Ynorm。

接着，Ynorm注入因果注意力层，得到Yatt, 然后经过Dropout 得到Yatt_drop

然后应用快捷连接，将 X + Yatt_drop 得到真正的Y。

 

Y作为新的X, 传入反馈神经网络。

同样，X会被归一化处理，处理成均值为0，方差为1的向量值 Ynorm。

接着，经过反馈神经网络，得到Yffn, 并经过Dropout 得到Yffn_drop。

最后再拼接 X 和Yffn_drop 得到Ｙ。　



todo：使用图来描述更加清晰  



```
class TransformerBlock(nn.Module):  # 需要继承nn.Module
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiAtention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            n_heads = cfg["n_heads"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.norm1 = NormLayer(cfg["emb_dim"])
        self.norm2 = NormLayer(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])  
        self.ffn = ForwardFeedLayer(cfg["drop_rate"])  
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)  
        x = self.att(x)    
        x = self.drop(x)
        x = shortcut + x
        
        shortcut = x
        x = self.norm2(x)  
        x = self.ffn(x)
        x = self.drop(x)
        
        return shortcut + x

		
```



#### 多头注意力机制　

#### 前馈神经网络

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



#### 层归一化



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

#### 残差连接

快捷连接(shortcut connection，又叫残差连接)是什么？

快捷连接一种在不同层之间增加连接的深度学习技术。快捷解决了反向传播时的梯度过小的问题 ——越靠前的神经网络层梯度越小。

如何实现快捷连接？ 

将当前神经网络的输入 x, 添加到输出之中。也就是x + 输出 = 新的输出。

当你在构建一个多层神经网络时，可以在不同的神经网络层之间添加快捷连接，从而避免梯度消失的问题。

在shortcut connect之前，可以执行dropout，这也是transformer block中的做法。



#### tranformer block代码实现 



```python
from previous_chapters import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块中的Shortcut连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        # 前馈块中的Shortcut连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        return x
```



### GPT类实现（数据输入，输出）



#### GPT核心有哪些组件？

```
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```



logits的 形状: (batch_size, seq_length, vocab_size)



forward函数

- embedding + pos- embedding 
- N个 transformer
- final-norm
- head-out



如何计算模型的参数量 

```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```



### 简单文本生成

使用贪婪解码生成文本。

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过了支持的长度，就对当前上下文进行截断
        # 例如，如果LLM只支持5个token，而上下文长度为10，
        # 那么只有最后5个token会被用作上下文

        idx_cond = idx[:, -context_size:]
        
        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]  

        # 通过softmax函数获得对应的概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 获取概率值最高的单词索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引添加到当前运行的上下文索引序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
```







模型是如何结束输出的？ 

​     模型什么时候终止，受到三个因素的影响。max_sequence , max_new_token, 模型主动输出停止符 `eos` .

在推理中，如果序列的长度达到了max_sequence， 模型就不会再输出。具体来说，

**（1）如果达到max_sequence_length（最大序列长度）**

- 这条序列**不能再生成新的token**，直接停止生成。
- 在实际实现中，这条序列会被**mask/pad**，即后续生成步骤中，这条序列不再被“激活”，只保留已生成内容。

**（2）如果提前遇到终止符（如EOS）**

- 这条序列也会停止生成，后续步骤用padding填充。







### GPT架构的进阶思考 

**元**

**反**



Logits每次都需要计算所有，导致会又很多浪费。因此 kv -cache诞生了

**空**



**什么是MOE模型？**
MOE，混合专家模型，是一种神经网络模型。由门控和多个专家网络构成。用于替换前馈神经网络。前馈神经网络是全部参数激活的，也叫做dense模型。而MOE则是先由门控来控制激活哪些专家，可以减少推理时的参数激活量。但MOE模型在训练难度上比Dense模型要高。 



**RMSＮorm是什么？** 

RMSNorm是一种基于均方根的归一化方法。GPT最开始使用的LayerNorm需要先减去均值，再除以方差，使得向量变成均值为０，方差为１的向量。RMSNorm不需要计算均值，也不使用方差来归一化。而是先将计算各项的平方和的均值，开根号，得到均方根。再用各项除以均方根，得到归一化后的值。

RMSＮorm的计算量比常规的LayerNorm要少，逐渐被主流的大模型采用，比如Qwen





### 小结



## 五、如何训练模型



token的编码和解码。

```python
import tiktoken
from previous_chapters import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 增加batch维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 去掉batch维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```





### 如何评估模型的输出？ 



使用输出结果和目标的距离来衡量。  





#### 交叉熵



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

#### 困惑度

困惑度是什么？ 

困惑度是对交叉熵进行指数计算的结果。 困惑度更有解释性，意味着模型在下一步中所不确定的词表的大小。

比如，当困惑度为10，那么意味着下一个词不确定是10中的哪一个。



#### 计算训练集和验证集的损失



```
from previous_chapters import create_dataloader_v1

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





### LLamaFactory的微调流程  





#### 数据集构建



#### 参数设置



#### 开始训练 





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



### 数据微调的进阶思考 

### 小结



## 七、附录



## 张量的基本操作

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

