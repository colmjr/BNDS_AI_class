#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:48:12 2026

@author: sunyifan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet18 (Pre-trained Transfer Learning) - Standard Version
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 自动检测设备：有CUDA用CUDA，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device} | 模式: ResNet18 迁移学习")

# 预训练模型通常需要 224x224 或 112x112 以获得更好效果
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载 CIFAR10 数据集
train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=transform), batch_size=64, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载预训练模型并修改最后全连接层以适配 CIFAR10 (10类)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) # 迁移学习建议用稍小的学习率

# 训练与评估
for epoch in range(2): # 预训练模型收敛极快
    model.train()
    correct_t, total_t = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_t += labels.size(0)
        correct_t += (predicted == labels).sum().item()
        
        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/{len(train_loader)}], Train Acc: {100*correct_t/total_t:.2f}%")

    model.eval()
    correct_v, total_v = 0, 0
    with torch.no_grad():
        for imgs, lbs in test_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            out = model(imgs)
            _, pred = torch.max(out.data, 1)
            total_v += lbs.size(0)
            correct_v += (pred == lbs).sum().item()
    print(f"==> Epoch {epoch+1} 测试集准确率: {100*correct_v/total_v:.2f}%")

# 结果可视化
model.eval()
# 获取一批测试数据
test_samples_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=10, shuffle=True)
images, labels = next(iter(test_samples_loader))
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(12, 3))
for i in range(10):
    ax = plt.subplot(1, 10, i+1)
    # ResNet 的反标准化较复杂，这里直接显示归一化后的原图
    img = images[i].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min()) # 归一化到 0-1 方便显示
    plt.imshow(img)
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f"{classes[predicted[i]]}", color=color)
    ax.axis('off')
plt.show()