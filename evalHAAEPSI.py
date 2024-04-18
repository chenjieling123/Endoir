# -*- coding:utf-8 -*-
# @FileName  :evalHAAEPSI.py
# @Time      :2024/4/15 9:26
# @Author    :CJL


from utils import *
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from datasets import SRDataset
from models import SRResNet, Generator
import time
import piq
from  haarPsi import *
# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 1  # 放大比例
ngpu = 2  # GPU数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == '__main__':

    # 测试集目录
    data_folder = "./data/upper-gi-tract-fbimage-new1_data/"  # "./data/"
    test_data_names = ["test"]

    # 预训练模型
    srgan_checkpoint = "./results/upper_gi_hyper_kvasir_all/srgan/checkpoint_srgan_1400.pth"
    # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))

    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'], False)
    # generator.load_state_dict(checkpoint['model'],False)

    # 多GPU测试
    if torch.cuda.is_available() and ngpu > 1:
        generator = nn.DataParallel(generator, device_ids=list(range(ngpu)))

    generator.eval()
    model = generator
    # srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    # srgan_generator.eval()
    # model = srgan_generator

    for test_data_name in test_data_names:
        print("\n数据集 %s:\n" % test_data_name)

        # 定制化数据加载器
        test_dataset = SRDataset(data_folder,
                                 split='test',
                                 crop_size=0,
                                 scaling_factor=1,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  pin_memory=True)

        # 记录每个样本 PSNR 和 SSIM值
        HaarPSIs = AverageMeter()

        # 记录测试时间
        start = time.time()
        HaarPSI = []

        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # 前向传播.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # 计算 PSNR 和 SSIM

                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel') # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel') # (w, h), in y-channel

                haarpsi=haar_psi_numpy(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy())
                #print(haarpsi[0])
                HaarPSIs.update(haarpsi,lr_imgs.size(0))
                HaarPSI.append(haarpsi)

        # 输出平均PSNR和SSIM
        # print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        # print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))

        HaarPSI.sort()
        print('Q1  {:.3f} '.format(HaarPSI[161]))
        print('Q2  {:.3f} '.format(HaarPSI[322]))
        print('Q3  {:.3f} '.format(HaarPSI[483]))
        print('HaarPSI  {mses.avg:.3f}'.format(mses=HaarPSIs))
        print('平均单张样本用时  {:.3f} 秒'.format((time.time() - start) / len(test_dataset)))

    print("\n")
