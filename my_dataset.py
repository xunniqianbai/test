import os
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import transforms
# 定义数据增强操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# 应用数据增强操作到数据集
# train_dataset = torchvision.datasets.ImageFolder(root='./LitsDataset/volume', transform=transform)
# class MyDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.file_list = os.listdir(root_dir)
#     def __len__(self):
#         return len(self.file_list)
#     def __getitem__(self, idx):
#         file_name = self.file_list[idx]
#         file_path = os.path.join(self.root_dir, file_name)
#         data = np.load(file_path)
#         # 对数据进行插值处理
#         data = [torch.tensor(d) for d in data]
#         data = [F.interpolate(d.unsqueeze(0), size=128, mode='nearest').squeeze(0) for d in data]
#         # 将数据包装成字典形式返回
#         data_dict = {'data': data}
#         return data_dict

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = os.listdir(root_dir)
        self.pad_size = (128, 128, 128)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.data_list[idx])
        data = np.load(data_path)
        depth, height, width = data.shape
        pad_depth = self.pad_size[0] - depth
        pad_height = self.pad_size[1] - height
        pad_width = self.pad_size[2] - width
        pad_widths = ((0, pad_depth), (0, pad_height), (0, pad_width))
        data = np.pad(data, pad_widths, mode='constant', constant_values=0)
        if self.transform:
            data = self.transform(data)
        return data, idx
    def __collate__(self, batch):
        # 获取 batch 中的最大尺寸
        max_size = tuple(max(s) for s in zip(*[data[0].shape for data in batch]))
        # 创建一个新的张量，用于存储所有数据
        batch_tensor = torch.zeros((len(batch),) + max_size)
        # 创建一个新的列表，用于存储所有数据的索引
        idx_list = []
        # 将数据填充到张量中
        for i, (data, idx) in enumerate(batch):
            pad_size = tuple((s - d) // 2 for s, d in zip(max_size, data.shape))
            batch_tensor[i] = F.pad(torch.from_numpy(data), pad_size, mode='constant', value=0)
            idx_list.append(idx)
        return batch_tensor, idx_list



#
    def load_data(data_path, batch_size, num_workers):
        dataset = MyDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# ###这里开始
#         train_dataset = MyDataset(root_dir='./LitsDataset/volume')
#       #  train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, collate_fn=train_dataset.__collate__)
#         for i in range(len(train_dataset)):
#             # check the size of each tensor in the i-th sample
#             tensor_sizes = [sample.size() for sample in train_dataset[i]]
#             if len(set(tensor_sizes)) > 1:
#                 print(f"Sample {i} has tensors of different sizes: {tensor_sizes}")
#                 ####这里删掉
        return dataloader




# class LiverDataset(Dataset):
#     def __init__(self, data_path, phase):
#         # your code to load data
#         self.phase = phase
#         self.data_path = data_path
#         self.data = self.load_data()
#         self.transforms = get_transforms(phase)
#         # check tensor sizes
#         for i in range(len(self.data)):
#             tensor_sizes = [sample.size() for sample in self.data[i]]
#             if len(set(tensor_sizes)) > 1:
#                 print(f"Sample {i} has tensors of different sizes: {tensor_sizes}")
#     def __getitem__(self, index):
#         # your code to get a sample
#         return sample
#     def __len__(self):
#         return len(self.data)
#
#
