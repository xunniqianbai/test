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
# import os
# import numpy as np
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset
# class LitsDataset(Dataset):
#     def __init__(self, data_dir, transforms=None):
#         data_dir="LitsDataset"
#         self.data_dir = data_dir
#         self.transforms = transforms
#         self.image_path_list = sorted([os.gpath.join(data_dir, "volume-" + str(i) + ".nii") for i in range(131)])
#         self.label_path_list = sorted([os.path.join(data_dir, "segmentation-" + str(i) + ".nii") for i in range(131)])
#     def __len__(self):
#         return len(self.image_path_list)
#     def __getitem__(self, index):
#         img_path = self.image_path_list[index]
#         label_path = self.label_path_list[index]
#         # 使用 nibabel 库读取 nii 数据
#         img_nii = nib.load(img_path)
#         img = np.array(img_nii.dataobj)
#         img = np.transpose(img, (2, 0, 1)) # 转换为 (C, H, W) 的形式
#         img = np.expand_dims(img, axis=0) # 添加 channel 维度
#         label_nii = nib.load(label_path)
#         label = np.array(label_nii.dataobj)
#         label = np.transpose(label, (2, 0, 1)) # 转换为 (C, H, W) 的形式
#         label = np.expand_dims(label, axis=0) # 添加 channel 维度
#         if self.transforms is not None:
#             img, label = self.transforms(img, label)
#         img_tensor = torch.from_numpy(img).float()
#         label_tensor = torch.from_numpy(label).float()
#         return img_tensor, label_tensor
import os
import numpy as np
import SimpleITK as sitk
import torch
import random
from torch.utils.data import Dataset

# class LitsDataset(Dataset):
#     def __init__(self, data_path, label_path, crop_size=(128, 128, 128), is_train=True):
#         self.crop_size = crop_size
#         self.is_train = is_train
#         self.data_path = data_path
#         self.label_path = label_path
#         self.data_files = os.listdir(self.data_path)
#         self.label_files = os.listdir(self.label_path)
#         self.data_files.sort()
#         self.label_files.sort()
#         assert len(self.data_files) == len(self.label_files), "data_files and label_files are not matched"
#     def __len__(self):
#         return len(self.data_files)
#     def __getitem__(self, index):
#         data = sitk.ReadImage(os.path.join(self.data_path, self.data_files[index]))
#         label = sitk.ReadImage(os.path.join(self.label_path, self.label_files[index]))
#         data_array = sitk.GetArrayFromImage(data)
#         label_array = sitk.GetArrayFromImage(label)
#         # crop
#         if self.is_train:
#             data_array, label_array = self.random_crop(data_array, label_array, self.crop_size)
#         else:
#             data_array, label_array = self.center_crop(data_array, label_array, self.crop_size)
#         # normalize
#         data_array = self.normalize(data_array)
#         # to tensor
#         data_tensor = torch.from_numpy(data_array).float()
#         label_tensor = torch.from_numpy(label_array).long()
#         return data_tensor, label_tensor
#     def center_crop(self, data, label, crop_size):
#         x, y, z = data.shape
#         startx = x//2-(crop_size[0]//2)
#         starty = y//2-(crop_size[1]//2)
#         startz = z//2-(crop_size[2]//2)
#         data = data[startx:startx+crop_size[0], starty:starty+crop_size[1], startz:startz+crop_size[2]]
#         label = label[startx:startx+crop_size[0], starty:starty+crop_size[1], startz:startz+crop_size[2]]
#         return data, label
#     def random_crop(self, data, label, crop_size):
#         x, y, z = data.shape
#         startx = np.random.randint(0, x-crop_size[0]+1)
#         starty = np.random.randint(0, y-crop_size[1]+1)
#         startz = np.random.randint(0, z-crop_size[2]+1)
#         data = data[startx:startx+crop_size[0], starty:starty+crop_size[1], startz:startz+crop_size[2]]
#         label = label[startx:startx+crop_size[0], starty:starty+crop_size[1], startz:startz+crop_size[2]]
#         return data, label
#     def normalize(self, image):
#         maxHU = 400.
#         minHU = -1000.
#         image[image > maxHU] = maxHU
#         image[image < minHU] = minHU
#         image = (image - minHU) / (maxHU - minHU)
#         return image
class LitsDataset(Dataset):
    def __init__(self, data_path, label_path, crop_size=(96, 96, 96), max_shape=(512, 512, 256), is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.data_path = data_path
        self.label_path = label_path
        self.data_files = os.listdir(self.data_path)
        self.label_files = os.listdir(self.label_path)
        self.data_files.sort()
        self.label_files.sort()
        assert len(self.data_files) == len(self.label_files), "data_files and label_files are not matched"
        self.max_shape = max_shape
        print(f"Data shape: {self.max_shape}")
        print(f"Crop size: {self.crop_size}")
    def __len__(self):
        return len(self.data_files)
    def __getitem__(self, index):
        data = sitk.ReadImage(os.path.join(self.data_path, self.data_files[index]))
        label = sitk.ReadImage(os.path.join(self.label_path, self.label_files[index]))
        data_array = sitk.GetArrayFromImage(data)
        label_array = sitk.GetArrayFromImage(label)
        # crop
        if self.is_train:
            data_array, label_array = self.random_crop(data_array, label_array, self.crop_size, self.max_shape)
        else:
            data_array, label_array = self.center_crop(data_array, label_array, self.crop_size)
        # normalize
        data_array = self.normalize(data_array)
        # to tensor
        data_tensor = torch.from_numpy(data_array).float()
        label_tensor = torch.from_numpy(label_array).long()
        return data_tensor, label_tensor
    def center_crop(self, data, label, crop_size):
        x, y, z = data.shape
        startx = x // 2 - (crop_size[0] // 2)
        starty = y // 2 - (crop_size[1] // 2)
        startz = z // 2 - (crop_size[2] // 2)
        data = data[startx:startx + crop_size[0],
               starty:starty + crop_size[1],
               startz:startz + crop_size[2]]
        label = label[startx:startx + crop_size[0],
                starty:starty + crop_size[1],
                startz:startz + crop_size[2]]
        return data, label
    # def random_crop(self, data, label, crop_size, max_shape):
    #     x, y, z = data.shape
    #     crop_x, crop_y, crop_z = crop_size
    #     max_x, max_y, max_z = max_shape
    #     max_startx = max_x - crop_x - 1
    #     max_starty = max_y - crop_y - 1
    #     max_startz = max_z - crop_z - 1
    #     if max_startx < 1 or max_starty < 1 or max_startz < 1:
    #         raise ValueError("Invalid crop size or max shape")
    #     startx = np.random.randint(crop_x // 2, min(x - crop_x // 2, max_startx) + 1)
    #     starty = np.random.randint(crop_y // 2, min(y - crop_y // 2, max_starty) + 1)
    #     startz = np.random.randint(crop_z // 2, min(z - crop_z // 2, max_startz) + 1)
    #     data = data[startx - crop_x // 2:startx + crop_x // 2,
    #            starty - crop_y // 2:starty + crop_y // 2,
    #            startz - crop_z // 2:startz + crop_z // 2]
    #     label = label[startx - crop_x // 2:startx + crop_x // 2,
    #             starty - crop_y // 2:starty + crop_y // 2,
    #             startz - crop_z // 2:startz + crop_z // 2]
    #     return data, label
    def random_crop(self, data_array, label_array, crop_size, max_shape):
        if data_array.ndim != 3:
            raise ValueError("Input data must be 3-dimensional")
        if label_array.ndim != 3:
            raise ValueError("Input label must be 3-dimensional")

        d, h, w = data_array.shape

        # select a random start point for cropping
        startd = random.randint(crop_size[0] // 2, max_shape[0] - crop_size[0] // 2)
        starth = random.randint(crop_size[1] // 2, max_shape[1] - crop_size[1] // 2)
        startw = random.randint(crop_size[2] // 2, max_shape[2] - crop_size[2] // 2)

        # crop the data and label arrays
        data_crop = data_array[startd - crop_size[0] // 2: startd + crop_size[0] // 2,
                    starth - crop_size[1] // 2: starth + crop_size[1] // 2,
                    startw - crop_size[2] // 2: startw + crop_size[2] // 2]
        label_crop = label_array[startd - crop_size[0] // 2: startd + crop_size[0] // 2,
                     starth - crop_size[1] // 2: starth + crop_size[1] // 2,
                     startw - crop_size[2] // 2: startw + crop_size[2] // 2]

        return data_crop, label_crop

    def normalize(self, image):
        maxHU = 400.
        minHU = -1000.
        image[image > maxHU] = maxHU
        image[image < minHU] = minHU
        image = (image - minHU) / (maxHU - minHU)
        return image