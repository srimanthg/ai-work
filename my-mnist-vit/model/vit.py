import torch
from model.patch_embedding import PatchEmbedding
from model.transformer import TransformerBlock
from model.mlp import MLPClassification


class VisionTransformer(torch.nn.Module):
    def __init__(self, num_classes, img_size, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        num_attn_heads = 4
        num_transformer_blocks = 2
        embed_dim = 32  # TODO change to 768
        num_mlp_hidden_size = 1024  # 768 > 1024 > 10 classes

        # Patch embedding (BS, embed_dim, num_patches)
        self.patch_embed = PatchEmbedding(
            in_channels=1,
            num_convs=embed_dim,
            patch_size=patch_size,
            num_patches=num_patches,
        )

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        # Transformer
        self.transformer_blocks = torch.nn.Sequential()
        for c in range(num_transformer_blocks):
            self.transformer_blocks.add_module(
                name=f"transformer{c}",
                module=TransformerBlock(
                    embed_dim=embed_dim,
                    num_attn_heads=num_attn_heads,
                    num_mlp_hidden_size=num_mlp_hidden_size,
                ),
            )

        # Classification head
        self.mlp = MLPClassification(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        bs = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer_blocks(x)
        x = self.mlp(x[:, 0])  # Only class tokens needed
        return x
