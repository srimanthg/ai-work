import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_points(x, y, line_slope=None, line_color="r"):
    plt.scatter(x.numpy(), y.numpy())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data Points")
    if line_slope is not None:
        x_vals = torch.tensor([x.min(), x.max()])
        y_vals = line_slope * x_vals
        plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.show()


"""
x = [-10, -5, 0, 5, 10]
y = [-30, -15, 0, 15, 30]  # y = 3 * x

weight (slope) = 0 (initialized)

After 1st epoch:
Predicted y = [0, 0, 0, 0, 0]
Loss = MSE([-30, -15, 0, 15, 30], [0, 0, 0, 0, 0]) = (900 + 225 + 0 + 225 + 900) / 5 = 450
Gradient

"""


def main():
    gt_slope = 3.0
    input_x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0]).view(-1, 1)
    input_y = gt_slope * input_x
    print(input_x)
    print(input_y)

    model = nn.Linear(1, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    print(f"Initial slope: {model.weight.item()}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(2):
        print(f"===== Epoch {epoch + 1} ======")
        optimizer.zero_grad()
        y_pred = model(input_x)
        loss = criterion(y_pred, input_y)
        print(
            f"slope: {model.weight.item()}, loss: {loss.item()}. GT slope: {gt_slope}"
        )
        print(f"Gradient before backward: {model.weight.grad}")
        loss.backward()
        print(f"Gradient after backward: {model.weight.grad}")
        optimizer.step()
        print(f"Weight after optimizer.step: {model.weight.item()}")


if __name__ == "__main__":
    main()
