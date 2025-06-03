# 监督微调(sft) — 让模型学会说话

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

