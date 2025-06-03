# 简单文本生成

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







