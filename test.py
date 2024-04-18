# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2023/11/17 9:21
# @Author    :CJL

from utils import *
from torch import nn
from arch.endoir_model import *
import time
from PIL import Image

# 测试图像
#imgPath = './data/upper-gi-tract-fbimage-new1/627.jpg'
imgPath='D:/CJL/GAN/data/rotate/627_270.jpg'
# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 1  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 预训练模型
    srgan_checkpoint = "./results/garnn-rawData/deepWBNet/checkpoint_deepWBNet16000.pth"
    #srgan_checkpoint = "./results/garnn-rawData/wbgan/checkpoint_wbgan_7000.pth"


    # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    #generator.load_state_dict(checkpoint['model'], False)
    generator.load_state_dict(checkpoint['generator'], False)


    generator.eval()
    model = generator

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    # 双线性上采样
    # Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    # Bicubic_img.save('./results/LR_160502_bicubic.jpg')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        #sr_img.save('./results/upper_gi_hyper_kvasir_all/srgan/14_result627.jpg')
        sr_img.save('D:/CJL/GAN/data/rotate/result_627_270.jpg')

    print('用时  {:.3f} 秒'.format(time.time() - start))

