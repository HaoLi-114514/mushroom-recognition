import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
from PIL import ImageFile

# 防止坏图报错
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ================= 配置区域 =================
# ================= 配置区域 =================
# 1. 关键：指向【去背景】的数据文件夹
# (这是你用 rembg 处理生成的那个文件夹)
DATA_DIR = 'cleaned_split_data'       

# 2. 关键：指向【去背景】的模型文件
# (这是你在 cleaned_split_data 上训练出来的那个准确率只有 60% 左右的模型)
# 请去文件夹里确认文件名，可能是 'mushroom_resnet18_cleaned.pth' 
# 如果你当时没改名覆盖了，可能需要重新训练一下 (见下文)
MODEL_PATH = 'mushroom_resnet18_cleaned.pth' 

NUM_CLASSES = 9
BATCH_SIZE = 32
# ===========================================
# ===========================================

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_test_data():
    """
    只加载 Test 集
    """
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader, test_dataset.classes

def load_trained_model():
    """
    重建模型架构并加载参数
    """
    print(f"Loading model from {MODEL_PATH}...")
    model = models.resnet18(pretrained=False) # 这里不需要下载预训练参数，因为我们要加载自己的
    
    # 必须重现训练时的修改：修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    # 关键：设置为评估模式！这会关闭 Dropout 和 BatchNorm 的随机性
    model.eval() 
    return model

def evaluate_model(model, test_loader, class_names):
    all_preds = []
    all_labels = []
    
    print("Starting evaluation on Test Set...")
    
    with torch.no_grad(): # 测试时不需要计算梯度，省显存
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 把结果存回 CPU 以便画图
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 打印文字报告
    print("\n" + "="*30)
    print("       TEST REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    # 1. 准备数据
    test_loader, class_names = load_test_data()
    print(f"Classes: {class_names}")
    
    # 2. 加载模型
    model = load_trained_model()
    
    # 3. 开始测试
    evaluate_model(model, test_loader, class_names)