import os
import sys
import time
import torch
import config
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil
import re


def init_model(model, device):
    for key, param in model.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = config.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)


def get_dataloader(type='train'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(config.cropsize),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(config.cropsize_val),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(config.cropsize_val),
            transforms.ToTensor(),
        ])
    }
    if type == 'train':
        train_images = datasets.ImageFolder(config.train_folder, data_transforms['train'])
        train_loader = DataLoader(train_images, batch_size=config.batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        # print(train_loader)
        return train_loader
    elif type == 'val':
        val_images = datasets.ImageFolder(config.val_folder, data_transforms['val'])
        val_loader = DataLoader(val_images, batch_size=config.val_batch_size,
                                shuffle=False, num_workers=0, drop_last=True)
        return val_loader
    elif type == 'test':
        test_images = datasets.ImageFolder(config.test_folder, data_transforms['test'])
        test_loader = DataLoader(test_images, batch_size=config.test_batch_size,
                                 shuffle=False, num_workers=0, drop_last=True)
        return test_loader
    else:
        raise ValueError('No such type!')


def log_starter(logger_name, root, level=logging.INFO, out='tofile'):
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    # print(root)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    if out == 'tofile':
        if not os.path.exists(root):
            os.makedirs(root)  # 创建目录
        path = os.path.join(root, logger_name + '-{}.log'.format(time.strftime("%Y%m%d-%H%M", time.localtime())))
        fh = logging.FileHandler(path, mode='w')
        fh.setFormatter(formatter)
        log.addHandler(fh)
    elif out == 'screen':
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)
    else:
        raise Exception("Invalid output option specified.")

    return log


def gauss_noise(shape, device):
    noise = torch.zeros(shape).to(device)  # shape形状的tensor
    for i in range(noise.shape[0]):  # shape[0]是noise的第一层特征，比如[[1,2,3],[4,5,6]],则shape[0]的值为2
        noise[i] = torch.randn(noise[i].shape).to(device)  # noise的，每一层生成一个标准正态分布

    return noise


# 网络参数 数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def model_structure(model, logger):
    blank = ' '
    message = '-' * 90 + '\n' + \
              '|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' + ' ' * 3 + 'number' + ' ' * 3 + '|\n' + \
              '-' * 90
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        message += '\n| {} | {} | {} |'.format(key, shape, str_num)
    message += '\n' + '-' * 90 + '\n' + 'The total number of parameters: ' + str(
        num_para) + '\n' + 'The parameters of Model {}: {:4f}M'.format(model._get_name(),
                                                                       num_para * type_size / 1000 / 1000) + '\n' + '-' * 90
    logger.info('\n' + message)


def plot_metrics(psnr_values, ssim_values, save_path):

    plt.figure(figsize=(12, 5))

    # 绘制PSNR
    plt.subplot(1, 2, 1)
    plt.plot(psnr_values, label='PSNR', color='blue', marker='o')
    plt.title('PSNR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.legend()

    # 绘制SSIM
    plt.subplot(1, 2, 2)
    plt.plot(ssim_values, label='SSIM', color='green', marker='o')
    plt.title('SSIM over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.legend()

    # 保存图表
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path)
    plt.close()


def load_from_hinet(model_path, model, optim, device):
    # 从HiNet的模型加载预训练模型
    # 加载模型状态字典
    state_dict = torch.load(model_path, map_location=device)

    # 处理 net 字典中的键名并排除含 'tmp_var' 的键
    net = state_dict.get('net', {})
    new_net = {}
    for key, value in net.items():
        if 'tmp_var' not in key:
            new_key = re.sub(r"module\.model\.inv\.(\d+)", lambda m: "model.inv.{}".format(int(m.group(1))), key)
            new_net[new_key] = value

    # 打印新的 keys，帮助识别问题
    print("New keys:", new_net.keys())

    # 更新模型参数
    try:
        model.load_state_dict(new_net)
    except RuntimeError as e:
        print('Failed to load model parameters:', e)

    # 尝试加载优化器状态
    try:
        optim.load_state_dict(state_dict['opt'])
    except KeyError:
        print('无法加载优化器状态')
    except Exception as e:
        print('Cannot load optimizer for some reason or other:', e)



def plot_and_save_images(cover=None, secret=None, steg=None, secret_rev=None, filename='output.png'):
    """Generates and saves a plot of given images with two rows and two columns."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 创建2行2列的子图
    images = [cover, steg, secret, secret_rev]
    titles = ['Cover', 'Steg', 'Secret', 'Secret Revealed']

    axs = axs.flatten()  # Flatten the axis array for easier iteration

    for ax, img, title in zip(axs, images, titles):
        if img is not None:
            if img.dim() == 4 and img.shape[0] > 0:
                img = img[0]  # 取批量中的第一张图
            img = img.permute(1, 2, 0).cpu().numpy()  # 转换为HWC格式

            # 根据数据类型调整图像数据
            if img.dtype == np.float32:  # 假设图像是float类型
                # 确保数据在 [0, 1] 范围内
                img = np.clip(img, 0, 1)
            elif img.dtype == np.uint8:
                # 对于uint8，确保数据已经在 [0, 255] 范围内，这里不需要调整
                pass

            ax.imshow(img, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def save_images(path, cover=None, secret=None, steg=None, secret_rev=None, epoch=None, id=None):
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the plot
    plot_filename = os.path.join(path, f'combined_image_{id if id is not None else "***"}.png')
    plot_and_save_images(cover, secret, steg, secret_rev, filename=plot_filename)

    # Now handle each image individually
    directories = {
        'cover': cover,
        'secret': secret,
        'steg': steg,
        'secret_rev': secret_rev
    }

    for key, img in directories.items():
        if img is not None:
            try:
                dir_path = os.path.join(path, key)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                img_filename = f"{key}{'_'+str(epoch)+'_' if epoch is not None else '_'}{id if id is not None else '***' :03d}.png"
                torchvision.utils.save_image(img, os.path.join(dir_path, img_filename))
            except Exception as e:
                print(f"Error saving {key}: {e}")


def delete_images(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted directory: {directory}")
    else:
        print("Directory does not exist.")