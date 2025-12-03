import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
from PIL import ImageFile

# 防止坏图报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 1. 搜索空间配置 (你可以随意改) =================
# 我们想尝试的学习率
LR_CANDIDATES = [0.001, 0.0005, 0.0001]

# 我们想尝试的 Batch Size (4080显存大，可以大胆点)
BATCH_SIZE_CANDIDATES = [32, 64, 128]

# 搜索阶段只跑几轮？(快速筛选)
SEARCH_EPOCHS = 6

# 最终选定参数后，跑几轮？(正式训练)
FINAL_EPOCHS = 25

# 数据路径 (用回原始数据)
DATA_DIR = 'split_data'
NUM_CLASSES = 9
# =============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_loaders(batch_size):
    # 保持最强的数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset

def train_session(lr, batch_size, num_epochs, session_name="Search"):
    """
    通用训练函数：既用于搜索，也用于最终训练
    """
    print(f"\n>>> [{session_name}] 启动: LR={lr}, Batch={batch_size}, Epochs={num_epochs}")
    
    # 1. 获取数据
    train_loader, val_loader, train_dataset = get_loaders(batch_size)
    
    # 2. 计算权重 (解决样本不平衡)
    class_counts = []
    for i in range(len(train_dataset.classes)):
        targets = torch.tensor(train_dataset.targets)
        count = (targets == i).sum().item()
        class_counts.append(count)
    weights = torch.FloatTensor([1.0/c for c in class_counts]).to(device)
    
    # 3. 初始化模型 (Transfer Learning)
    model = models.resnet18(weights='IMAGENET1K_V1')
    # 搜索阶段可以冻结层加快速度，最终训练也可以选择解冻(这里保持统一冻结策略)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    # 4. 定义优化器
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    # 5. 训练循环
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # 打印简单进度
        print(f"    Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
        
    print(f">>> [{session_name}] 结束。最佳准确率: {best_val_acc:.4f}")
    return best_val_acc, model

if __name__ == '__main__':
    print("=======================================")
    print("   PHASE 1: 自动参数搜索 (Grid Search)")
    print("=======================================")
    
    best_combo_acc = 0.0
    best_lr = 0.001
    best_bs = 32
    
    # 双重循环遍历所有组合
    for lr in LR_CANDIDATES:
        for bs in BATCH_SIZE_CANDIDATES:
            # 跑一个小型的训练 session
            acc, _ = train_session(lr, bs, SEARCH_EPOCHS, session_name="Searching")
            
            if acc > best_combo_acc:
                best_combo_acc = acc
                best_lr = lr
                best_bs = bs
                print(f"    *** 发现新王者！LR={lr}, Batch={bs} (Acc={acc:.4f}) ***")
    
    print("\n" + "="*40)
    print(f"   搜索完成！最强参数组合: LR={best_lr}, Batch={best_bs}")
    print("   PHASE 2: 使用最强参数进行最终长训练")
    print("="*40)
    
    # 使用刚才找到的 best_lr 和 best_bs 跑完全程
    final_acc, final_model = train_session(best_lr, best_bs, FINAL_EPOCHS, session_name="Final Training")
    
    # 保存模型
    save_name = f"mushroom_resnet18_auto_tuned_lr{best_lr}_bs{best_bs}.pth"
    torch.save(final_model.state_dict(), save_name)
    print(f"\n全自动流程结束！最终模型已保存为: {save_name}")
    print(f"最终准确率: {final_acc:.4f}")