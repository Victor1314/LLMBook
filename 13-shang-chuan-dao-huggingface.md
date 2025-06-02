# 上传到 Huggingface

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

