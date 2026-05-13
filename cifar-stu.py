#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 22:23:09 2026

@author: sunyifan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 修改 root 路径：确保 ./data 目录下有 cifar-10-python.tar.gz
train_set = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

print(f"数据加载成功，共有训练记录：{len(train_set)}条")

# 3. 定义自定义 CNN 结构
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
        #
        )
        self.classifier = nn.Sequential(
        #   
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCNN().to(device)

# 4. 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练演示
print("开始从零训练自定义模型 (Scratch)...")
for epoch in range(3):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # --- 新增：计算当前 Batch 的准确率 ---
        if (i+1) % 100 == 0:
            # 获取预测结果（概率最大的索引）
            _, predicted = torch.max(outputs.data, 1)
            # 计算正确个数
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            
            print(f"Epoch [{epoch+1}/3], Step [{i+1}/782], Loss: {loss.item():.4f}, Batch Accuracy: {accuracy:.2f}%")

print("自定义模型训练演示完成。\n")
