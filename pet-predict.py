import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from train import ShapeUNet # 从 train.py 导入结构

def student_predict(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 1. 加载模型
    model = ShapeUNet().to(device)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
    except:
        print("❌ 未找到 best_model.pth，请先运行训练脚本。")
        return

    # 2. 处理图片
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy() > 0.5

    # 4. 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.imshow(raw_img.resize((128,128))); plt.title("Your Image"); plt.axis('off')
    plt.subplot(122); plt.imshow(mask, cmap='gray'); plt.title("AI Segmentation"); plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 替换成学生自己的图片文件名
    student_predict('test.jpg')