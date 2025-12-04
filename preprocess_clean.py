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
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # 随机打乱
        random.shuffle(images)
        
        # 计算划分数量
        count = len(images)
        train_idx = int(count * SPLIT_RATIO[0])
        val_idx = int(count * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
        
        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:]
        }

        print(f"   正在处理类别 '{cls}' (共 {count} 张)...")

        for split_name, file_list in splits.items():
            for file_name in file_list:
                src = os.path.join(cls_path, file_name)
                
                # 注意：必须保存为 .png 才能保留去背景后的透明通道
                new_name = os.path.splitext(file_name)[0] + ".png"
                dst = os.path.join(TARGET_DIR, split_name, cls, new_name)
                
                try:
                    with open(src, 'rb') as i:
                        with open(dst, 'wb') as o:
                            input_img = i.read()
                            
                            # === 核心步骤：AI 智能去背景 ===
                            subject = remove(input_img) 
                            
                            o.write(subject)
                    
                    total_images += 1
                    # 每处理 100 张提示一下进度
                    if total_images % 100 == 0:
                        print(f"      已累计处理 {total_images} 张图片...")
                        
                except Exception as e:
                    print(f"      [跳过坏图] {file_name}: {e}")

    print("-" * 30)
    print(f">>> 处理完成！所有去背景图片已保存在: {TARGET_DIR}/")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"发生未知错误: {e}")
    input("按回车键退出...")
