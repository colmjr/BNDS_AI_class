#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insurance Cost Prediction with PyTorch
Enhanced: train/test loss visualization + MAE
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('insurance.csv')

# 2. 特征工程
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X = preprocessor.fit_transform(df)
y = df['charges'].values.reshape(-1, 1)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)  # 标准化目标变量

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. PyTorch Dataset
class InsuranceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = InsuranceDataset(X_train, y_train)
test_dataset = InsuranceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. 定义模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = RegressionModel(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. 训练 + 记录训练/测试 Loss
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试 Loss
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.astype(np.float32))
        y_pred_test = model(X_test_tensor)
        test_loss = criterion(y_pred_test, torch.tensor(y_test.astype(np.float32))).item()
        test_losses.append(test_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}')

# 7. 最终评估（反标准化）
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_pred_scaled = model(X_test_tensor).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # MSE 和 R2
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 新增：MAE
    mae = mean_absolute_error(y_true, y_pred)

    print(f'\nFinal Test MSE: {mse:.2f}, R2 Score: {r2:.4f}, MAE: {mae:.2f}')

# 8. 可视化训练 & 测试 Loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss (MSE, scaled)')
plt.plot(test_losses, label='Test Loss (MSE, scaled)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (scaled)')
plt.title('Training & Test Loss Curves')
plt.legend()
plt.show()

# 9. 可视化预测结果
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.show()
