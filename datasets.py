# -*- coding:utf-8 -*-
# @FileName  :aberration_recovery.py
# @Time      :2023/10/31 10:09
# @Author    :CJL

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms


class SRDataset(Dataset):
    """
    数据集加载器
    """

    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        """
        :参数 data_folder: # Json数据文件所在文件夹路径
        :参数 split: 'train' 或者 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸  （实际训练时不会用原图进行放大，而是截取原图的一个子块进行放大）
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        :参数 test_data_name: 如果是评估阶段，则需要给出具体的待评估数据集名称，例如 "Set14"
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("请提供测试数据集名称!")
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        # 如果是测试，则不需要对图像的长宽作限定
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "裁剪尺寸不能被放大比例整除!"
        '''
        if self.split == 'train':
            with open(os.path.join(data_folder, 'HR_train_images.json'), 'r') as j:
                self.HR_images = json.load(j)
            with open(os.path.join(data_folder, 'LR_train_images.json'), 'r') as j:
                self.LR_images = json.load(j)
        else:
            with open(os.path.join(data_folder, 'HR_test_images.json'), 'r') as j:
                self.HR_images = json.load(j)
            with open(os.path.join(data_folder, 'LR_test_images.json'), 'r') as j:
                self.LR_images = json.load(j)
        '''

        # 读取图像路径
        if self.split == 'train':
            with open(os.path.join(data_folder, 'HR_train_images.json'), 'r') as j:
                self.HR_images = json.load(j)
            with open(os.path.join(data_folder, 'LR_train_images.json'), 'r') as j:
                self.LR_images = json.load(j)
        else:
            with open(os.path.join(data_folder, 'HR_test_images.json'), 'r') as j:
                self.HR_images = json.load(j)
            with open(os.path.join(data_folder, 'LR_test_images.json'), 'r') as j:
                self.LR_images = json.load(j)
#            with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
#                self.images = json.load(j)

        # 数据处理方式
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.

        :参数 i: 图像检索号
        :返回: 返回第i个低分辨率和高分辨率的图像对
        """
        # 读取图像
        img1 = Image.open(self.HR_images[i], mode='r')
        img1 = img1.convert('RGB')
        img2 = Image.open(self.LR_images[i], mode='r')
        img2 = img2.convert('RGB')
        if img1.width <= 96 or img1.height <= 96:
            print(self.HR_images[i], img1.width, img1.height)
        if img2.width <= 96 or img2.height <= 96:
            print(self.LR_images[i], img2.width, img2.height)
#        lr_img,hr_img=self.LR_images[i],self.HR_images[i]
        lr_img, hr_img = self.transform(img1,img2)

        return lr_img, hr_img

    def __len__(self):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.

        :返回: 加载的图像总数
        """
        return len(self.HR_images)
