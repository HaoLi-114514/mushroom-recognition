import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from PIL import ImageFile

# 防止坏图报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 配置区域 =================
DATA_DIR = 'split_data'       # 用回含有背景的原始数据 (Context很重要)
BATCH_SIZE = 64               # 4080显存大，从零训练可以开大一点Batch
LEARNING_RATE = 0.001
NUM_EPOCHS = 30               # <--- 从零训练需要更多轮次
NUM_CLASSES = 9 
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_class_weights(dataset):
    print("计算类别权重...")
    class_counts = []
    for i in range(len(dataset.classes)):
        targets = torch.tensor(dataset.targets)
        count = (targets == i).sum().item()
        class_counts.append(count)
    
    weights = [1.0 / c for c in class_counts]
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(device)

def get_data_loaders():
    # 数据增强 (从零训练非常依赖这个)
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, train_dataset

# ==========================================
# 核心修改：模型初始化 (From Scratch)
# ==========================================
def initialize_model_scratch():
    print("正在初始化 ResNet18 (随机权重，无预训练)...")
    
    # weights=None 表示随机初始化
    model = models.resnet18(weights=None) 
    
    # 注意：这里我们删除了 param.requires_grad = False 的循环
    # 因为我们要训练所有的层！
    
    # 修改输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    print("开始从零训练 (这可能会比迁移学习慢)...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    plt.title("Accuracy (From Scratch)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title("Loss (From Scratch)")
    plt.legend()
    plt.savefig("training_curves_scratch.png")
    print("曲线图已保存为 training_curves_scratch.png")

if __name__ == '__main__':
    train_loader, val_loader, train_dataset = get_data_loaders()
    
    # 依然使用权重平衡，因为数据不平衡的问题还在
    class_weights = compute_class_weights(train_dataset)
    
    # 初始化空模型
    model = initialize_model_scratch()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 注意：从零训练时，通常需要更新模型的所有参数，所以这里用 model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model, train_acc, val_acc, train_loss, val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    plot_curves(train_acc, val_acc, train_loss, val_loss)
    torch.save(model.state_dict(), "mushroom_resnet18_scratch.pth") # 保存为scratch版本
    print("从零训练模型已保存！")