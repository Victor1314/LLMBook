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
