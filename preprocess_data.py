import os
import shutil
import random
from PIL import Image, ImageFile
from rembg import remove  # <--- 新加入的库

# 防止报错截断图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 配置区域 =================
SOURCE_DIR = 'raw_data' 
TARGET_DIR = 'cleaned_split_data' # <--- 改个新名字，别覆盖原来的
SPLIT_RATIO = [0.7, 0.15, 0.15] 
# ===========================================

def main():
    print(">>> 1. Script started. Checking directories...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Directory '{SOURCE_DIR}' not found!")
        return

    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f">>> 2. Found {len(classes)} classes.")

    # 创建文件夹
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    total_images = 0
    print(">>> 3. Starting CLEANING and SPLITTING... (This will take time!)")
    
    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        files = os.listdir(cls_path)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.shuffle(images)
        count = len(images)
        train_idx = int(count * SPLIT_RATIO[0])
        val_idx = int(count * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
        
        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:]
        }

        print(f"   Processing class '{cls}' ({count} images)...")

        for split_name, file_list in splits.items():
            for file_name in file_list:
                src = os.path.join(cls_path, file_name)
                # 注意：保存为 .png 以保留透明背景，或者转为黑色背景 jpg
                # 这里我们统一保存为 png
                new_name = os.path.splitext(file_name)[0] + ".png"
                dst = os.path.join(TARGET_DIR, split_name, cls, new_name)
                
                try:
                    # =========================================
                    # 核心修改：这里不再是 copy，而是 remove background
                    # =========================================
                    with open(src, 'rb') as i:
                        with open(dst, 'wb') as o:
                            input_img = i.read()
                            subject = remove(input_img) # 智能去背景
                            o.write(subject)
                    
                    # 只有成功了才计数
                    total_images += 1
                    if total_images % 100 == 0:
                        print(f"      Processed {total_images} images so far...")
                        
                except Exception as e:
                    print(f"Skipping error image {file_name}: {e}")

    print("-" * 30)
    print(f">>> DONE! Cleaned images saved to: {TARGET_DIR}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    input("Press Enter to exit...")