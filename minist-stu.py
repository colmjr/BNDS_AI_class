"""
实验：操控 VAE 隐变量的某一维，观察 MNIST 数字的连续变化
操作：运行代码，你会看到一行数字图片，从左到右逐渐变化。
     修改下面 CONTROL_DIM 和 VALUES 两个变量，观察不同维度的控制效果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ========== 你可以修改这里 ==========
CONTROL_DIM = 0               # 想要控制的隐变量维度（0~15）
VALUES = [-3, -2, -1, 0, 1, 2, 3]   # 该维度的取值序列
# ==================================

# 超参数（已预训练好，不需要重新训练）
latent_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 VAE 模型（必须包含 forward，与训练时一致）
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 加载预训练好的模型参数（如果没有会自动训练，约1-2分钟）
model = VAE(latent_dim).to(device)
try:
    model.load_state_dict(torch.load("mnist_vae.pth", map_location=device))
    print("加载预训练模型成功。")
except:
    print("未找到预训练模型，开始训练（约1-2分钟）...")
    train_data = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            BCE = F.binary_cross_entropy(recon, images, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.2f}")
    torch.save(model.state_dict(), "mnist_vae.pth")
    print("训练完成，模型已保存。")

# 生成控制实验图像
model.eval()
with torch.no_grad():
    z = torch.zeros((len(VALUES), latent_dim)).to(device)
    z[:, CONTROL_DIM] = torch.tensor(VALUES).to(device)
    generated = model.decode(z).cpu()
    grid = vutils.make_grid(generated, nrow=len(VALUES), normalize=True, padding=2)
    plt.figure(figsize=(14, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Effect of changing latent dimension {CONTROL_DIM} (values: {VALUES})")
    plt.show()

print("提示：尝试修改代码开头的 CONTROL_DIM 和 VALUES，观察不同维度的控制效果。")