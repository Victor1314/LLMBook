# 创建Space，show出你的模型

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







