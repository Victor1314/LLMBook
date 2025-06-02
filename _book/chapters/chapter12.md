# 创建键权重矩阵，形状和 W_query 相同，同样设置 requires_grad=False
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

