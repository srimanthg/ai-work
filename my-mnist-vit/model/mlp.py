import torch
import torchvision


class MLPClassification(torch.nn.Module):
    def __init__(self, embed_dim, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.linear = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x = x[:, 0]  # Need only CLS token
        x = self.layer_norm(x)
        x = self.linear(x)
        return x
