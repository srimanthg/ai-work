import torch
import torchvision


class TransformerBlock(torch.nn.Module):

    def __init__(self, embed_dim, num_attn_heads, num_mlp_hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_attn_heads, batch_first=True
        )
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, num_mlp_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(num_mlp_hidden_size, embed_dim),
        )

    def forward(self, x):
        residual_1 = x
        x = self.layer_norm1(x)
        x = self.multihead_attn(x, x, x)[0]
        x = x + residual_1
        residual_2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual_2
        return x
