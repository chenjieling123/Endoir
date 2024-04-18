# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2023/10/31 10:09
# @Author    :CJL

from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as FT
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
        参数 min_size: 图像宽、高的最小容忍值
        参数 output_folder: 最终生成的文件列表,json格式
    """
    #hyper kvasir中共有118,209张图片,其中118,000为训练集,209为测试集
    train_images = list()
    for d in train_folders:
        d = d + '/upper-gi-tract-image/'
        train_name = "upper_gi_hyper_kvasir_1"
        for i in range(1, 2575):
            img_path = d + str(i) + '.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name, len(train_images)))
    with open(os.path.join(output_folder, train_name + '_HR_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    train_images = list()
    for d in train_folders:
        d = d + '/upper-gi-tract-fbimage-new1/'
        train_name = "upper_gi_hyper_kvasir_1"
        for i in range(1, 2575):
            img_path = d + str(i) + '.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name, len(train_images)))
    with open(os.path.join(output_folder, train_name + '_LR_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/upper-gi-tract-image/'
        test_name = "upper_gi_hyper_kvasir_1"
        for i in range(2575, 3219):
            img_path = d + str(i) + '.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
          (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_HR_test_images.json'), 'w') as j:
        json.dump(test_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/upper-gi-tract-fbimage-new1/'
        test_name = "upper_gi_hyper_kvasir_1"
        for i in range(2575, 3219):
            img_path = d + str(i) + '.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
          (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_LR_test_images.json'), 'w') as j:
        json.dump(test_images, j)
    '''
    print("\n正在创建文件列表... 请耐心等待.\n")
    train_images = list()
    train_images1 = list()
    test_images = list()
    test_images1 = list()
    for d in train_folders:
        d1 = d + '/LR'
        d = d + '/HR'

        train_name = d.split("/")[-1]
        train_name1 = d1.split("/")[-1]
        test_name = d.split("/")[-1]
        test_name1 = d1.split("/")[-1]
        lens=len(os.listdir(d))
        j=0
        for i in os.listdir(d):
            if j<0.88*lens:
                img_path = os.path.join(d, i)
                i1 = "L" + i[1:]
                img_path1 = os.path.join(d1, i1)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    train_images.append(img_path)
                img1 = Image.open(img_path1, mode='r')
                if img1.width >= min_size and img1.height >= min_size:
                    train_images1.append(img_path1)
            else:
                img_path = os.path.join(d, i)
                i1 = "L" + i[1:]
                img_path1 = os.path.join(d1, i1)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    test_images.append(img_path)
                if img1.width >= min_size and img1.height >= min_size:
                    test_images1.append(img_path1)
            j+=1


    print("在训练集%s中共有 %d 张图像\n" % (train_name, len(train_images)))
    with open(os.path.join(output_folder, train_name + '_train_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, train_name1 + '_train_images.json'), 'w') as j:
        json.dump(train_images1, j)
    print("在测试集 %s 中共有 %d 张图像\n" %
          (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, test_name1 + '_test_images.json'), 'w') as j:
        json.dump(test_images1, j)
    '''
    '''
    print("\n正在创建文件列表... 请耐心等待.\n")
    train_images = list()
    train_images1=list()
    for d in train_folders:
        d1 = d + '/LR'
        d = d + '/HR'

        train_name = d.split("/")[-1]
        train_name1 = d1.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            i1="L"+i[1:]
            img_path1=os.path.join(d1,i1)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
            img1 = Image.open(img_path1, mode='r')
            if img1.width >= min_size and img1.height >= min_size:
                train_images1.append(img_path1)
    print("在训练集%s中共有 %d 张图像\n" % (train_name, len(train_images)))
    with open(os.path.join(output_folder, train_name + '_train_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, train_name1 + '_train_images.json'), 'w') as j:
        json.dump(train_images1, j)



    test_images = list()
    test_images1=list()
    for d in test_folders:
        d1 = d + '/LR'
        d = d + '/HR'

        test_name = d.split("/")[-1]
        test_name1=d1.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            i1 = "L" + i[1:]
            img_path1 = os.path.join(d1, i1)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
            if img1.width >= min_size and img1.height >= min_size:
                test_images1.append(img_path1)
    print("在测试集 %s 中共有 %d 张图像\n" %
          (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, test_name1 + '_test_images.json'), 'w') as j:
        json.dump(test_images1, j)
    '''


    '''

    train_images = list()
    for d in train_folders:
        d=d+'/upper-gi-tract-image/'
        train_name = "upper_gi_hyper_kvasir"
        for i in range(1,2253):
            img_path = d+str(i)+'.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name,len(train_images)))
    with open(os.path.join(output_folder, train_name+'_HR_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    train_images = list()
    for d in train_folders:
        d = d + '/upper-gi-tract-fbimage-new/'
        train_name = "upper_gi_hyper_kvasir"
        for i in range(1,2253):
            img_path = d+str(i)+'.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name,len(train_images)))
    with open(os.path.join(output_folder, train_name+'_LR_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/upper-gi-tract-image/'
        test_name ="upper_gi_hyper_kvasir"
        for i in range(2253,3219):
            img_path = d+str(i)+'.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
            (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_HR_test_images.json'),'w') as j:
        json.dump(test_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/upper-gi-tract-fbimage-new/'
        test_name = "upper_gi_hyper_kvasir"
        for i in range(2253,3219):
            img_path = d+str(i)+'.jpg'
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
            (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_LR_test_images.json'), 'w') as j:
        json.dump(test_images, j)
    '''

    '''

    train_images = list()
    for d in train_folders:
        d=d+'/HR'
        train_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name,len(train_images)))
    with open(os.path.join(output_folder, train_name+'_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    train_images = list()
    for d in train_folders:
        d = d + '/LR'
        train_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("在训练集%s中共有 %d 张图像\n" % (train_name,len(train_images)))
    with open(os.path.join(output_folder, train_name+'_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/HR'
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
            (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_test_images.json'),'w') as j:
        json.dump(test_images, j)

    test_images = list()
    for d in test_folders:
        d = d + '/LR'
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
    print("在测试集 %s 中共有 %d 张图像\n" %
            (test_name, len(test_images)))
    with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
        json.dump(test_images, j)
    '''

    print("生成完毕。训练集和测试集文件列表已保存在 %s 下\n" % output_folder)


def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]' 
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]' 
                   (3) '[-1, 1]' 
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img1,img2):
        """
        :参数 img1,img2: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img1.width - self.crop_size)
            top = random.randint(1, img1.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img1.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img1.width % self.scaling_factor
            y_remainder = img1.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img1.width - x_remainder)
            bottom = top + (img1.height - y_remainder)
            hr_img = img1.crop((left, top, right, bottom))

        if self.split == 'train':
            # 从对应的低分辨率图片中随机裁剪一个子块作为低分辨率图像
            left = random.randint(1, img2.width - self.crop_size)
            top = random.randint(1, img2.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            lr_img = img2.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img2.width % self.scaling_factor
            y_remainder = img2.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img1.width - x_remainder)
            bottom = top + (img1.height - y_remainder)
            lr_img = img2.crop((left, top, right, bottom))

        # 下采样（双三次差值）
        lr_img = lr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)


        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    保存训练结果.

    :参数 state: 逐项预保存内容
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :参数 optimizer: 需要调整的优化器
    :参数 shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """

    print("\n调整学习率.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("新的学习率为 %f\n" % (optimizer.param_groups[0]['lr'], ))
