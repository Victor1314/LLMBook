# 创建查询权重矩阵，形状为 (d_in, d_out)，并且设置 requires_grad=False 表示这些权重在训练过程中不会更新
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

