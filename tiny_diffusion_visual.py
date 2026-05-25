import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# ===============================
# 0. 创建输出目录
# ===============================
os.makedirs("samples", exist_ok=True)

# ===============================
# 1. 超参数（学生可以改！）
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 128
epochs = 5
lr = 1e-3
timesteps = 100

# ===============================
# 2. 数据集
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===============================
# 3. 模型
# ===============================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / timesteps
        x = torch.cat([x, t], dim=1)
        return self.net(x)

model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===============================
# 4. 噪声调度
# ===============================
betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

# ===============================
# 5. 采样函数（关键！！！）
# ===============================
def sample(model, save_steps=False, name="sample"):
    model.eval()
    with torch.no_grad():
        x = torch.randn((1, 28*28)).to(device)

        images = []

        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t], device=device)

            pred_noise = model(x, t_tensor)

            alpha = alphas[t]
            alpha_hat_t = alpha_hat[t]
            beta = betas[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise

            # ⭐ 保存过程（关键教学点）
            if save_steps and t % 10 == 0:
                img = x.view(28,28).cpu()
                images.append(img)

        final_img = x.view(28,28).cpu()

    return final_img, images


# ===============================
# 6. 训练 + 可视化
# ===============================
loss_history = []

for epoch in range(epochs):
    model.train()

    for imgs, _ in loader:
        imgs = imgs.view(imgs.size(0), -1).to(device)

        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
        noise = torch.randn_like(imgs)

        alpha_t = alpha_hat[t].unsqueeze(1)

        noisy_imgs = torch.sqrt(alpha_t) * imgs + torch.sqrt(1 - alpha_t) * noise

        pred_noise = model(noisy_imgs, t)

        loss = nn.MSELoss()(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch {epoch} Loss: {loss.item()}")

    # ⭐ 每个epoch生成一张图
    img, _ = sample(model)
    plt.imshow(img, cmap="gray")
    plt.title(f"Epoch {epoch}")
    plt.axis("off")
    plt.savefig(f"samples/epoch_{epoch}.png")
    plt.close()

# ===============================
# 7. 最终生成过程可视化
# ===============================
final_img, process_imgs = sample(model, save_steps=True)

# 展示生成过程
plt.figure(figsize=(10,2))
for i, img in enumerate(process_imgs):
    plt.subplot(1, len(process_imgs), i+1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
plt.suptitle("Denoising Process")
plt.show()

# ===============================
# 8. Loss曲线
# ===============================
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()