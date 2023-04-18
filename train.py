import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from LITSDataset import LitsDataset
from UNet3D_ViT_GCN import UNet3D_ViT_GCN



from my_dataset import MyDataset




train_dataset = MyDataset(root_dir='./LitsDataset/volume')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, collate_fn=train_dataset.__collate__)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# 计算mean IoU指标

def mean_iou(outputs, targets):
    smooth = 1e-6
    outputs = torch.argmax(outputs, dim=1).flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()
    intersection = np.sum(outputs * targets)
    union = np.sum(outputs) + np.sum(targets) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
# 定义超参数
batch_size = 4
lr = 0.001
epochs = 200
fold = 5 # 5折交叉验证
# 定义数据集和数据加载器
dataset = LitsDataset(data_path='LitsDataset/volume', label_path='LitsDataset/segmentation', crop_size=(128, 128, 128), is_train=True)
indices = np.arange(len(dataset))
np.random.shuffle(indices)
fold_indices = np.array_split(indices, fold)
# 开始交叉验证
for i in range(fold):
    print(f"Fold {i+1} of {fold}")
    train_indices = np.concatenate(fold_indices[:i] + fold_indices[i+1:])
    val_indices = fold_indices[i]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 定义模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet3D_ViT_GCN(num_classes=2).to(device)
    model = UNet3D_ViT_GCN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 开始训练
    best_val_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0
        for j, data in enumerate(train_loader):
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_iou += mean_iou(outputs, targets)
        train_loss = running_loss/len(train_loader)
        train_iou = running_iou/len(train_loader)
        print(f"Fold {i+1} of {fold} Epoch {epoch+1}/{epochs} Train Loss: {train_loss} Train Mean IoU: {train_iou}")
        # 验证模型
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_iou = 0.0
            for j, data in enumerate(val_loader):
                inputs, targets = data['input'].to(device), data['target'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                running_iou += mean_iou(outputs, targets)
            val_loss = running_loss/len(val_loader)
            val_iou = running_iou/len(val_loader)
            print(f"Fold {i+1} of {fold} Epoch {epoch+1}/{epochs} Val Loss: {val_loss} Val Mean IoU: {val_iou}")
            # 保存模型权重
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
                torch.save(model.state_dict(), f'saved_models/best_model_fold_{i+1}_epoch_{epoch+1}.pt')
            # 将训练集和验证集上的损失函数、mean IoU指标保存到results.txt中
            with open('results.txt', 'a') as f:
                f.write(f"Fold {i+1} of {fold} Epoch {epoch+1}/{epochs} Train Loss: {train_loss} Train Mean IoU: {train_iou} Val Loss: {val_loss} Val Mean IoU: {val_iou}\n")
        model.train()