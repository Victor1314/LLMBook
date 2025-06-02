# 预训练(pretrain) —— 让模型学习海量知识

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



