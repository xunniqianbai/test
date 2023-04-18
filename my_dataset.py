import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path)
        # 对数据进行插值处理
        data = [torch.tensor(d) for d in data]
        data = [F.interpolate(d.unsqueeze(0), size=128, mode='nearest').squeeze(0) for d in data]
        # 将数据包装成字典形式返回
        data_dict = {'data': data}
        return data_dict