# 定义自注意力模块的第二个版本
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        # 调用父类构造函数
        super().__init__()
        # 设置输出维度
        self.d_out = d_out
        # 初始化查询、键和值的线性层，可以选择是否包含偏置项
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 使用线性层将输入 x 投影到查询、键和值空间
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 计算注意力分数（未归一化）
        attn_scores = queries @ keys.T
        
        # 使用 softmax 函数和缩放因子归一化注意力分数
        # 注意这里的 dim=1，表示沿着键向量的维度进行归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        # 使用归一化的注意力权重和值向量计算上下文向量
        context_vec = attn_weights @ values
        return context_vec

