import torch


class PatchEmbedding(torch.nn.Module):

    def __init__(self, in_channels, num_convs, patch_size, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_convs,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = x.flatten(2)
        x = torch.transpose(x, 1, 2)
        return x
