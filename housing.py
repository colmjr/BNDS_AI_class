import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =========================
# 1. 加载数据
# =========================
X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2. 标准化
# =========================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# =========================
# 3. 线性回归（Baseline）
# =========================
lr = LinearRegression()
lr.fit(X_train_std, y_train)

y_pred_lr = lr.predict(X_test_std)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Linear Regression Test MSE: {mse_lr:.4f}")

# =========================
# 4. PyTorch 数据
# =========================
X_train_t = torch.tensor(X_train_std, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_std, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# =========================
# 5. 简单神经网络
# =========================
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =========================
# 6. 训练神经网络
# =========================
epochs = 500
train_losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

# =========================
# 7. 神经网络测试 MSE
# =========================
with torch.no_grad():
    y_pred_nn = model(X_test_t)
    mse_nn = criterion(y_pred_nn, y_test_t).item()

print(f"Neural Network Test MSE:   {mse_nn:.4f}")

# =========================
# 8. 可视化
# =========================
plt.figure(figsize=(12, 5))

# ---- (1) MSE 对比 ----
plt.subplot(1, 2, 1)
plt.bar(["Linear Regression", "Neural Network"], [mse_lr, mse_nn])
plt.ylabel("MSE")
plt.title("Test MSE Comparison")

# ---- (2) 神经网络训练曲线 ----
plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training Curve")

plt.tight_layout()
plt.show()
