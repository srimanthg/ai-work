import torch
import torchvision
from model.vit import VisionTransformer


def main():
    # Settings
    num_classes = 10
    batch_size = 16
    img_size = 28
    patch_size = 7
    lr = 0.001

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

        step = 0
        for images, values in train_dl:
            optimizer.zero_grad()

            pred = vit(images)
            loss = criteria(pred, values)
            loss.backward()

            optimizer.step()
            step += 1

            # Accuracy

            if step % 100 == 0:
                with torch.no_grad():
                    total_images = 0
                    correct_images = 0
                    for v_images, v_values in val_dl:
                        pred_v_values = vit(v_images)
                        pred_v_values = torch.argmax(pred_v_values, dim=1)
                        correct_images += (pred_v_values == v_values).sum()
                        total_images += v_values.shape[0]
                    print(
                        f"Epoch {epoch}, Step {step}: {correct_images}/{total_images} correct ({100*correct_images/total_images:0.2f} %)"
                    )


if __name__ == "__main__":
    main()
