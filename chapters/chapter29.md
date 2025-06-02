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


