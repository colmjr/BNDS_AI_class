import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# --- 1. 配置参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果是CPU运行，建议将 imsize 设为 128 以保证演示流畅
imsize = 512 if torch.cuda.is_available() else 128  

# VGG19 预训练模型要求的标准化参数
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# --- 2. 图像处理工具 ---
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    orig_size = image.size  # 记录原始尺寸 (W, H)
    
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
    ])
    
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), orig_size

def tensor_to_pil(tensor, orig_size=None):
    """将训练中的 Tensor 转回 PIL 图像，并可选恢复原始尺寸"""
    img = tensor.cpu().clone().detach().squeeze(0)
    
    # 反标准化 (De-normalization)
    inv_normalize = transforms.Normalize(
        mean=-cnn_normalization_mean / cnn_normalization_std,
        std=1 / cnn_normalization_std
    )
    img = inv_normalize(img)
    img = torch.clamp(img, 0, 1) # 限制像素值在 [0,1]
    
    pil_img = transforms.ToPILImage()(img)
    
    if orig_size is not None:
        # 恢复到最原始的图片长宽比
        pil_img = pil_img.resize(orig_size, Image.LANCZOS)
    return pil_img

# --- 3. 构建模型 ---
class StyleModel(nn.Module):
    def __init__(self):
        super(StyleModel, self).__init__()
        # 选取经典的 VGG19 层用于提取内容和风格
        self.chosen_features = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.chosen_features:
                features[self.chosen_features[name]] = x
        return features

def get_gram_matrix(tensor):
    _, d, h, w = tensor.size()
    features = tensor.view(d, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(d * h * w)

# --- 4. 主程序 ---
def run_style_transfer(content_path, style_path, num_steps=500):
    content_img, content_orig_size = image_loader(content_path)
    style_img, _ = image_loader(style_path)
    
    # 迭代的目标是这张 input_img，初始为内容图
    input_img = content_img.clone().requires_grad_(True)
    
    model = StyleModel()
    optimizer = optim.LBFGS([input_img]) # LBFGS 对风格迁移效果极佳
    
    content_weight = 1
    style_weight = 1e6 # 风格权重通常设得非常大

    step = [0]
    while step[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            input_features = model(input_img)
            content_features = model(content_img)
            style_features = model(style_img)

            c_loss = nn.functional.mse_loss(input_features['conv4_1'], content_features['conv4_1'])
            
            s_loss = 0
            for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
                target_gram = get_gram_matrix(style_features[layer])
                input_gram = get_gram_matrix(input_features[layer])
                s_loss += nn.functional.mse_loss(input_gram, target_gram)

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()
            
            step[0] += 1
            if step[0] % 50 == 0:
                print(f"Iteration {step[0]}, Loss: {total_loss.item():.4f}")
            return total_loss

        optimizer.step(closure)

        # --- 阶段性显示图片 ---
        if step[0] % 100 == 0 or step[0] == 1:
            plt.figure()
            # 这里的显示仅用于进度查看，不恢复原图大小以保持响应速度
            plt.imshow(tensor_to_pil(input_img))
            plt.title(f"Step {step[0]}")
            plt.axis('off')
            plt.show(block=False) 
            plt.pause(0.5)

    # --- 最终结果处理 ---
    print("生成结束，正在保存高分辨率结果...")
    final_img = tensor_to_pil(input_img, content_orig_size) # 恢复原始大小
    final_img.save("final_result.png")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(final_img)
    plt.title("Final Result (Original Resolution)")
    plt.axis('off')
    plt.show() # 阻塞显示，直到手动关闭所有窗口

# 运行 (确保目录下有这两个文件)
if __name__ == "__main__":
    # 请手动放置 content.jpg 和 style.jpg
    run_style_transfer("content.jpg", "style.jpg")