import torch
import torch.nn as nn
import torch.optim as optim

# 1) Define the custom layer
class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 trainable parameters, initialized from N(0,1)
        self.weights = nn.Parameter(torch.randn(8))

    def forward(self, x):
        # elementwise multiply input by our 8 weights
        return x * self.weights

# 2) Create a tiny “dataset” where y = true_weights * x
true_weights = torch.arange(1, 9, dtype=torch.float32)  # [1,2,3,...,8]
def make_batch(batch_size=16):
    # random inputs
    x = torch.randn(batch_size, 8)
    # targets constructed so the layer should learn true_weights
    y = x * true_weights
    return x, y

# 3) Instantiate layer, loss, optimizer
layer = CustomLayer()
optimizer = optim.SGD(layer.parameters(), lr=0.1)
loss_fn   = nn.MSELoss()

# 4) Inspect initial weights
print("Initial weights:", layer.weights.data.numpy())

# 5) Training loop
for epoch in range(1, 201):
    x_batch, y_batch = make_batch(batch_size=32)
    optimizer.zero_grad()
    y_pred = layer(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | loss: {loss.item():.6f}")

# 6) Inspect learned weights
print("Learned weights:", layer.weights.data.numpy())
print("Target weights :", true_weights.numpy())
