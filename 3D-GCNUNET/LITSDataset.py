# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import nibabel as nib
#
#
# class LITSDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(os.path.join(root_dir, "images")))
#         self.label_files = sorted(os.listdir(os.path.join(root_dir, "labels")))
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         # 读取图像和标签数据
#         image_path = os.path.join(self.root_dir, "images", self.image_files[idx])
#         label_path = os.path.join(self.root_dir, "labels", self.label_files[idx])
#         image = nib.load(image_path).get_fdata()
#         label = nib.load(label_path).get_fdata()
#         # 将数据转换为PyTorch的张量格式
#         image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
#         label = torch.from_numpy(np.expand_dims(label, axis=0)).float()
#         # 数据增强
#         if self.transform:
#             image, label = self.transform(image, label)
#         # 返回图像和标签数据
#         return image, label
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
class LitsDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_path_list = sorted([os.path.join(data_dir, "volume-" + str(i) + ".nii") for i in range(131)])
        self.label_path_list = sorted([os.path.join(data_dir, "segmentation-" + str(i) + ".nii") for i in range(131)])
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        label_path = self.label_path_list[index]
        # 使用 nibabel 库读取 nii 数据
        img_nii = nib.load(img_path)
        img = np.array(img_nii.dataobj)
        img = np.transpose(img, (2, 0, 1)) # 转换为 (C, H, W) 的形式
        img = np.expand_dims(img, axis=0) # 添加 channel 维度
        label_nii = nib.load(label_path)
        label = np.array(label_nii.dataobj)
        label = np.transpose(label, (2, 0, 1)) # 转换为 (C, H, W) 的形式
        label = np.expand_dims(label, axis=0) # 添加 channel 维度
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).float()
        return img_tensor, label_tensor