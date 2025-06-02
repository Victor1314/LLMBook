# 💻 环境准备


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


