# 遍历输入序列中的每个元素
for i, x_i in enumerate(inputs):
    # 对于当前的输入元素 x_i，再次遍历整个输入序列
    for j, x_j in enumerate(inputs):
        # 计算 x_i 和 x_j 的点积，作为注意力分数，并存储在 attn_scores 矩阵的对应位置
        attn_scores[i, j] = torch.dot(x_i, x_j)

