import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# --- 用户操作：图像加载与预处理 ---
# 1. 请将您要测试的图片上传到运行环境，并替换下面的 "your_image.jpg" 为您的图片路径
# 2. 如果您的图片是彩色图，下面的代码会将其转换为灰度图进行演示。
#    如果希望处理彩色图像，卷积核和apply_custom_kernel函数需要相应调整以处理多通道。
try:
    # 请替换为您的图片路径
    image_path = "房屋.jpg" 
    img = Image.open(image_path).convert("L") # 转换为灰度图
    print(f"成功加载图像: {image_path}")
except FileNotFoundError:
    print(f"错误：找不到图像文件 {image_path}。请确保文件路径正确或上传文件。")
    print("将使用一个默认的示例图像进行演示。")
    # 创建一个默认的简单示例图像 (如果用户图片加载失败)
    img_data_default = np.array(
        [[220, 220, 220, 100, 100, 100],
         [220, 220, 220, 100, 100, 100],
         [220, 220, 220, 100, 100, 100],
         [50, 50, 50, 180, 180, 180],
         [50, 50, 50, 180, 180, 180],
         [50, 50, 50, 180, 180, 180]], dtype=np.float32)
    img = Image.fromarray(img_data_default)

# 图像预处理: 转换为Tensor
preprocess = transforms.Compose([
    transforms.Resize((128, 128)), # 可以调整大小以适应不同图片
    transforms.ToTensor()
])
img_tensor = preprocess(img).unsqueeze(0) # (1, 1, H, W) - 添加batch和channel维度

print(f"图像张量形状: {img_tensor.shape}")

def apply_custom_kernel(image_tensor, kernel_np, kernel_name="Custom Kernel"):
    """
    对图像张量应用自定义卷积核。
    image_tensor: 输入图像张量，形状 (1, 1, H, W) - 灰度图
    kernel_np: numpy数组表示的卷积核
    kernel_name: 卷积核的名称，用于绘图
    """
    # 确保kernel_np是2D的
    if kernel_np.ndim != 2:
        raise ValueError("卷积核必须是2D的 numpy 数组。")

    kernel_h, kernel_w = kernel_np.shape
    kernel_tensor = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).float()

    # 动态计算padding以保持输出尺寸与输入大致相同 (对于stride=1)
    # padding = (kernel_size - 1) // 2
    padding_h = (kernel_h - 1) // 2
    padding_w = (kernel_w - 1) // 2

    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, 
                           kernel_size=(kernel_h, kernel_w), 
                           padding=(padding_h, padding_w),
                           bias=False)

    conv_layer.weight.data = kernel_tensor
    conv_layer.weight.requires_grad = False

    with torch.no_grad():
        output_tensor = conv_layer(image_tensor)

    print(f"应用 {kernel_name} 后输出形状: {output_tensor.shape}")
    return output_tensor

# --- 定义各种卷积核 ---

# 1. 横向边缘检测核 (Prewitt)
kernel_horizontal_edge = np.array([[-1, -1, -1],
                                   [ 0,  0,  0],
                                   [ 1,  1,  1]], dtype=np.float32)

# 2. 纵向边缘检测核 (Prewitt)
kernel_vertical_edge = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)

# 3. 均值模糊核 (3x3)
kernel_mean_blur_3x3 = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], dtype=np.float32) / 9.0

# 4. 近似高斯模糊核 (3x3)
kernel_gaussian_approx_3x3 = np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]], dtype=np.float32) / 16.0

# --- 用户操作：在此处定义您的5x5高斯模糊核 ---
# 请替换下面的 None 为您的 5x5 numpy 数组核
# 例如:
kernel_gaussian_5x5_user = np.array([
    [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
    [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
    [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
    [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
    [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]
], dtype=np.float32)
 # (确保归一化，即所有元素和为1)
# kernel_gaussian_5x5_user = None 
# --- 用户操作结束 ---


# --- 应用卷积核 ---
output_horizontal = apply_custom_kernel(img_tensor, kernel_horizontal_edge, "Horizontal Edge (Prewitt)")
output_vertical = apply_custom_kernel(img_tensor, kernel_vertical_edge, "Vertical Edge (Prewitt)")
output_mean_blur_3x3 = apply_custom_kernel(img_tensor, kernel_mean_blur_3x3, "Mean Blur 3x3")
# output_gaussian_approx_3x3 = apply_custom_kernel(img_tensor, kernel_gaussian_approx_3x3, "Gaussian Blur 3x3 (Approx)")

outputs_to_plot = [
    (img_tensor, 'Original Image'),
    (output_horizontal, 'Horizontal Edges'),
    (output_vertical, 'Vertical Edges'),
    (output_mean_blur_3x3, 'Mean Blur 3x3'),
    # (output_gaussian_approx_3x3, 'Gaussian Blur 3x3 (Approx)')
]

if kernel_gaussian_5x5_user is not None:
    if isinstance(kernel_gaussian_5x5_user, np.ndarray) and kernel_gaussian_5x5_user.shape == (5,5):
        # 确保用户提供的核是归一化的 (可选，但推荐用于模糊)
        if not np.isclose(np.sum(kernel_gaussian_5x5_user), 1.0) and np.sum(kernel_gaussian_5x5_user) != 0:
            print("警告: 用户提供的5x5高斯核未归一化 (元素和不为1)。建议进行归一化以获得预期的模糊效果。")
        output_gaussian_5x5_user = apply_custom_kernel(img_tensor, kernel_gaussian_5x5_user, "User 5x5 Gaussian Blur")
        outputs_to_plot.append((output_gaussian_5x5_user, 'User 5x5 Gaussian Blur'))
    else:
        print("错误: 用户提供的 kernel_gaussian_5x5_user 不是一个 5x5的 numpy 数组，将不进行处理。")


# --- 可视化结果 ---
num_images = len(outputs_to_plot)
fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
if num_images == 1: # 如果只有一个图像，axs不是列表
    axs = [axs] 

for i, (tensor_data, title) in enumerate(outputs_to_plot):
    axs[i].imshow(tensor_data.squeeze().numpy(), cmap='gray')
    axs[i].set_title(title)
    axs[i].axis('off')

plt.tight_layout()
plt.show() # 在实际Python环境中运行时，取消此行注释以显示图像
# 在某些交互式环境 (如Jupyter Notebook) 中，图像可能会自动显示
# 如果在脚本中运行，确保调用 plt.show()