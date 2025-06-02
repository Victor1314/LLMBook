# 我们创建的掩码形状应该和注意力权重矩阵的形状一致，以一一对应
block_size = attn_scores.shape[0]
