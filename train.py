import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import ImageFile

# 防止坏图报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 配置区域 =================
DATA_DIR = 'split_data'       # <--- 确认这里是指向原始数据
BATCH_SIZE = 32               # <--- 你的报错是因为缺少了这一行！
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_CLASSES = 9 
# ===========================================

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ... (后面的代码保持不变) ...

def compute_class_weights(dataset):
    """
    自动计算类别权重，解决 'Lactarius 偏见' 问题
    数量越少的类别，权重越大。
    """
    print("正在计算类别权重以平衡数据...")
    class_counts = []
    for i in range(len(dataset.classes)):
        # 统计每个类别的图片数量
        targets = torch.tensor(dataset.targets)
        count = (targets == i).sum().item()
        class_counts.append(count)
        
    print(f"各类别数量: {dict(zip(dataset.classes, class_counts))}")
    
    # 简单的倒数权重计算
    weights = [1.0 / c for c in class_counts]
    
    # 归一化，让权重总和不要太大
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * len(class_counts)
    
    print(f"计算出的权重: {weights}")
    return weights.to(device)

def get_data_loaders():
    # === 数据增强升级 ===
    # 针对 Agaricus 等难识别类别，加入颜色抖动
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # 蘑菇可以倒过来
        transforms.RandomRotation(20),
        # 颜色抖动：改变亮度对比度，迫使模型关注形状而非背景颜色
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 检查文件夹是否存在
    train_path = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到文件夹: {train_path}。请先运行 preprocess_data.py")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, train_dataset

def initialize_model():
    model = models.resnet18(pretrained=True)
    
    # 冻结层
    for param in model.parameters():
        param.requires_grad = False
        
    # 修改输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    print("开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # 这里会自动应用 Class Weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def plot_curves(train_acc, val_acc, train_loss, val_loss):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.savefig("training_curves_cleaned.png")
    print("曲线图已保存为 training_curves_cleaned.png")

if __name__ == '__main__':
    # 1. 加载数据
    train_loader, val_loader, train_dataset = get_data_loaders()
    
    # 2. 计算权重 (解决样本不平衡)
    class_weights = compute_class_weights(train_dataset)
    
    # 3. 初始化模型
    model = initialize_model()
    
    # 4. 定义 Loss (带权重!)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # 5. 训练
    model, train_acc, val_acc, train_loss, val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # 6. 保存结果
    plot_curves(train_acc, val_acc, train_loss, val_loss)
    torch.save(model.state_dict(), "mushroom_resnet18_final.pth") # <--- 改个新名字
    print("最终模型已保存！")