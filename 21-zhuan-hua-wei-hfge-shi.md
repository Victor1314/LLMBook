# 转化为HF格式

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



