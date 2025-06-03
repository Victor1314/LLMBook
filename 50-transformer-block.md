# transformer block



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



