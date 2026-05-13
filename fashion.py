#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 09:17:35 2026

@author: sunyifan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. 配置超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 Fashion-MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 定义可配置的神经网络
class ExperimentalNet(nn.Module):
    def __init__(self, use_bn=False, dropout_rate=0.0):
        super(ExperimentalNet, self).__init__()
        self.use_bn = use_bn
        
        # 卷积层 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        
        # 卷积层 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一组 卷积 + BN + ReLU + Pool
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # 第二组 卷积 + BN + ReLU + Pool
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接 + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. 训练函数
def train_model(use_bn, dropout_rate, use_adam=True):
    model = ExperimentalNet(use_bn, dropout_rate).to(DEVICE)
    
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': []}
    
    print(f"\n开始训练: BN={use_bn}, Dropout={dropout_rate}, Optimizer={'Adam' if use_adam else 'SGD'}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return history

# 5. 执行对比实验
# 实验 A: 基准 
hist_baseline = train_model(use_bn=False, dropout_rate=0.0, use_adam=False)

# 实验 B: 加入 BN + Adam
hist_optimized = train_model(use_bn=True, dropout_rate=0.5, use_adam=True)

# 6. 可视化对比结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_baseline['loss'], label='Baseline (SGD, No BN)')
plt.plot(hist_optimized['loss'], label='Optimized (Adam, BN, Dropout)')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_baseline['accuracy'], label='Baseline')
plt.plot(hist_optimized['accuracy'], label='Optimized')
plt.title('Training Accuracy')
plt.legend()
plt.show()