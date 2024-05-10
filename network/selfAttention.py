import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # 调整 x 的维度以符合 MultiheadAttention 的要求：[seq_len, batch_size, embedding_dim]
        # 假设 x 的原始维度为 [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)  # 将高和宽合并作为序列长度
        x = x.permute(2, 0, 1)  # 转置为 [seq_len, batch_size, channels]

        # 应用注意力机制
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)

        return attn_output
