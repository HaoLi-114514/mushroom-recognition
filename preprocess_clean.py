import os
import shutil
import random
from PIL import Image, ImageFile
from rembg import remove  # 必须先 pip install rembg

# 防止遇到下载不完整的图片报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 配置区域 =================
SOURCE_DIR = 'raw_data'               # 原始图片文件夹
TARGET_DIR = 'cleaned_split_data'     # 处理后保存的文件夹 (去背景)
SPLIT_RATIO = [0.7, 0.15, 0.15]       # Train / Val / Test 比例
# ===========================================

def main():
    print(">>> 1. 脚本启动，正在检查文件夹...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 找不到 '{SOURCE_DIR}' 文件夹。请确保它在当前目录下。")
        return

    # 获取所有类别
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f">>> 2. 发现 {len(classes)} 个类别: {classes}")

    # 创建目标文件夹结构
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    total_images = 0
    print(">>> 3. 开始去背景并划分数据 (这需要一些时间，请耐心等待)...")
    
    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        files = os.listdir(cls_path)
        # 筛选图片文件
        images = [f for f in files if f.lower().endswith(('.png', '.