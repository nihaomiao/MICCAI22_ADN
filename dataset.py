import torch
import os
import numpy as np
from torch.utils import data
import nibabel as nib
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random


class StrokeTrain3D(data.Dataset):
    def __init__(self, data_dir, train_txt, num_ct,
                 is_jitter=True, is_mirror=True, is_rotate=True,
                 vertical=True, resize=False, use_denoise=False, use_norm=False, gamma=64./255):
        super(StrokeTrain3D, self).__init__()
        self.gamma = gamma
        self.vertical = vertical
        self.resize = resize
        self.num_ct = num_ct
        self.use_denoise = use_denoise
        self.use_norm = use_norm
        with open(train_txt, "r") as f:
            self.name_list = f.read().split()
        self.ct_list = [os.path.join(data_dir, x, "CT.nii") for x in self.name_list]
        self.msk_list = [os.path.join(data_dir, x, "mask.nii") for x in self.name_list]
        self.is_jitter = is_jitter
        self.is_mirror = is_mirror
        self.is_rotate = is_rotate
        self.jitter_transform = transforms.ColorJitter(brightness=64. / 255, contrast=0.25, saturation=0.25, hue=0.04)

    def __len__(self):
        return int(self.num_ct)

    def __getitem__(self, index):
        index = index % len(self.ct_list)
        name = self.name_list[index]
        ct_data = nib.load(self.ct_list[index]).get_fdata()
        msk_data = nib.load(self.msk_list[index]).get_fdata()

        ct_data = np.nan_to_num(ct_data, 0.)
        msk_data = np.nan_to_num(msk_data, 0.)

        chosen_ct_tensor = transforms.ToTensor()(ct_data)
        chosen_msk_tensor = transforms.ToTensor()(msk_data)
        chosen_ct_tensor = chosen_ct_tensor/255.0
        chosen_msk_tensor = chosen_msk_tensor/5.0
        chosen_ct_tensor = chosen_ct_tensor.unsqueeze(dim=1)
        chosen_msk_tensor = chosen_msk_tensor.unsqueeze(dim=1)
        # change ct and msk to be vertical
        if self.vertical:
            chosen_ct_tensor = F.rotate(chosen_ct_tensor, -90)
            chosen_msk_tensor = F.rotate(chosen_msk_tensor, -90)
        # resize ct and msk to be (256, 256)
        if self.resize:
            chosen_ct_tensor = F.resize(chosen_ct_tensor, size=[256, 256])
            chosen_msk_tensor = F.resize(chosen_msk_tensor, size=[256, 256])
        # color jitter
        if self.is_jitter:
            gamma_f = random.uniform(max(0, 1 - self.gamma), 1 + self.gamma)
            chosen_ct_tensor = F.adjust_gamma(chosen_ct_tensor, gamma_f)
        # mirror
        if self.is_mirror:
            if random.random() < 0.5:
                if self.vertical:
                    chosen_ct_tensor = F.hflip(chosen_ct_tensor)
                    chosen_msk_tensor = F.hflip(chosen_msk_tensor)
                else:
                    chosen_ct_tensor = F.vflip(chosen_ct_tensor)
                    chosen_msk_tensor = F.vflip(chosen_msk_tensor)
        # rotate
        if self.is_rotate:
            angle = random.uniform(-10, +10)
            chosen_ct_tensor = F.rotate(chosen_ct_tensor, angle)
            chosen_msk_tensor = F.rotate(chosen_msk_tensor, angle)
        chosen_msk_tensor = chosen_msk_tensor*5.0
        chosen_ct_tensor = chosen_ct_tensor.permute(1, 0, 2, 3)
        chosen_msk_tensor = chosen_msk_tensor.permute(1, 0, 2, 3)

        return chosen_ct_tensor.type(torch.float32), chosen_msk_tensor.squeeze(dim=0).type(torch.float32), name


class StrokeTest3D(data.Dataset):
    def __init__(self, data_dir, test_txt, vertical=True, resize=False):
        super(StrokeTest3D, self).__init__()
        self.vertical = vertical
        self.resize = resize
        with open(test_txt, "r") as f:
            self.name_list = f.read().split()
        self.ct_list = [os.path.join(data_dir, x, "CT.nii") for x in self.name_list]
        self.msk_list = [os.path.join(data_dir, x, "mask.nii") for x in self.name_list]

    def __len__(self):
        return len(self.ct_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        chosen_ct = nib.load(self.ct_list[index]).get_fdata()
        chosen_msk = nib.load(self.msk_list[index]).get_fdata()
        chosen_ct_tensor = transforms.ToTensor()(chosen_ct)
        chosen_msk_tensor = transforms.ToTensor()(chosen_msk)
        chosen_ct_tensor = chosen_ct_tensor/255.0
        chosen_msk_tensor = chosen_msk_tensor/5.0
        chosen_ct_tensor = chosen_ct_tensor.unsqueeze(dim=1)
        chosen_msk_tensor = chosen_msk_tensor.unsqueeze(dim=1)
        # change ct and msk to be vertical
        if self.vertical:
            chosen_ct_tensor = F.rotate(chosen_ct_tensor, -90)
            chosen_msk_tensor = F.rotate(chosen_msk_tensor, -90)
        # resize ct and msk to be (256, 256)
        if self.resize:
            chosen_ct_tensor = F.resize(chosen_ct_tensor, size=[256, 256])
            chosen_msk_tensor = F.resize(chosen_msk_tensor, size=[256, 256])
        chosen_msk_tensor = chosen_msk_tensor*5.0
        chosen_ct_tensor = chosen_ct_tensor.permute(1, 0, 2, 3)
        chosen_msk_tensor = chosen_msk_tensor.permute(1, 0, 2, 3)
        return chosen_ct_tensor.type(torch.float32), chosen_msk_tensor.squeeze(dim=0).type(torch.float32), name
