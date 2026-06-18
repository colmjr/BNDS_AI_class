#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立预测脚本：加载训练好的 ShapeUNet 模型，对单张图片进行宠物分割
兼容检查点格式（包含 model_state_dict）
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -------------------- 模型定义（与训练脚本保持一致）--------------------
class ShapeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = self._block(3, 32)
        self.e2 = self._block(32, 64)
        self.e3 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.b = self._block(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = self._block(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = self._block(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(32, 1, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.b(self.pool(s3))
        x = torch.cat([self.u3(b), s3], dim=1)
        x = self.d3(x)
        x = torch.cat([self.u2(x), s2], dim=1)
        x = self.d2(x)
        x = torch.cat([self.u1(x), s1], dim=1)
        x = self.d1(x)
        return torch.sigmoid(self.final(x))
# -------------------------------------------------------------------

def student_predict(img_path, weight_path='best_model_optimized.pth'):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"使用设备: {device}")

    # 加载模型
    model = ShapeUNet().to(device)
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件 {weight_path} 不存在，请先训练模型。")
        return
    
    try:
        # 加载检查点（由于 PyTorch 2.6+ 默认 weights_only=True，这里显式设为 False 以兼容旧格式）
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        
        # 判断是检查点字典还是纯模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✅ 从检查点中提取 model_state_dict")
        else:
            state_dict = checkpoint
            print("✅ 直接加载模型权重")
        
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(img_path):
        print(f"❌ 图片文件 {img_path} 不存在")
        return

    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy() > 0.5

    # 可视化（保持原图比例）
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(raw_img)
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(122)
    # 将 mask 放大到原图尺寸
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(raw_img.size, Image.NEAREST)
    plt.imshow(mask_img, cmap='gray')
    plt.title("分割结果")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 请确保图片文件存在，并修改文件名或路径
    student_predict('test.jpg')