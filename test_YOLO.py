import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. 加载模型
model = YOLO('runs/detect/license_plate_exp/weights/best.pt')

# 2. 推理 (注意：这里强行把 imgsz 设为 640，看看能不能救回来)
img_path = 'test2.png'
results = model.predict(source=img_path, imgsz=640, conf=0.1) # 降低阈值到0.1，宁可错杀不可放过

# 3. 获取结果并绘图
res = results[0]
img_bgr = res.orig_img  # 原始图像
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # 转换为 RGB 格式

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
ax = plt.gca()

# 检查是否检测到目标
if len(res.boxes) == 0:
    plt.title("No license plate detected. Try retraining with imgsz=640!")
else:
    # 遍历所有检测到的框
    for box in res.boxes:
        # 获取坐标 (x1, y1, x2, y2)
        coords = box.xyxy[0].tolist()
        conf = box.conf[0].item() # 置信度
        
        # 绘制矩形框 (左上角 x, y, 宽, 高)
        rect = plt.Rectangle((coords[0], coords[1]), 
                             coords[2] - coords[0], 
                             coords[3] - coords[1], 
                             fill=False, color='red', linewidth=3)
        ax.add_patch(rect)
        
        # 添加置icn度文字
        ax.text(coords[0], coords[1] - 10, f'Plate: {conf:.2f}', 
                bbox=dict(facecolor='red', alpha=0.5), color='white')
    
    plt.title(f"Detected {len(res.boxes)} plate(s)")

plt.axis('off') # 隐藏坐标轴
plt.show()