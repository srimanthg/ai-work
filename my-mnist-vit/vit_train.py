import torch
import torchvision
from model.vit import VisionTransformer


def main():
    # Settings
    num_classes = 10
    batch_size = 16
    img_size = 28
    patch_size = 7
    lr = 0.01

    # Datasets
    ds_transforms = torchvision.transforms.Compose(
        transforms=[torchvision.transforms.ToTensor()]
    )
    train_ds = torchvision.datasets.MNIST(
        root="data/mnist", train=True, download=True, transform=ds_transforms
    )
    val_ds = torchvision.datasets.MNIST(
        root="data/mnist", train=False, download=True, transform=ds_transforms
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    vit = VisionTransformer(
        num_classes=num_classes, img_size=img_size, patch_size=patch_size
    )
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters(), lr=lr)

    for epoch in range(100):
        total_images = 0
        correct_images = 0
        for images, values in train_dl:
            optimizer.zero_grad()

            pred = vit(images)
            loss = criteria(pred, values)
            loss.backward()

            optimizer.step()

            # Accuracy
            pred_values = pred.argmax(dim=1)
            correct_images += (pred_values == values).sum()
            total_images += values.shape[0]
        print(
            f"Epoch {epoch}: {correct_images}/{total_images} correct ({100*correct_images/total_images}%)"
        )


if __name__ == "__main__":
    main()
