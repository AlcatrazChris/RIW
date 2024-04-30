import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:

    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(
                        n_x, k_x, size_x)
    return filters


def dct_coeff(n, k, N):

    return np.cos(np.pi / N * (n + 0.5) * k)


def idct_coeff(n, k, N):
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):

    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    for i, j in index_order[:keep_count]:
        mask[i, j] = 1

    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[: image_shape[0], : image_shape[1]]


def rgb2yuv(image_rgb, image_yuv_out):
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :] + 0.587 * image_rgb[:, 1, :, :] + 0.114 * image_rgb[:, 2,:, :]
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :] + -0.28886 * image_rgb[:, 1, :, :] + 0.436 * image_rgb[:, 2, :,:]
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :] + -0.51499 * image_rgb[:, 1, :, :] + -0.10001 * image_rgb[:, 2, :,:]


def yuv2rgb(image_yuv, image_rgb_out):
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :] + 1.13983 * image_yuv[:, 2, :, :]
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :] + -0.39465 * image_yuv[:, 1, :, :] + -0.58060 * image_yuv[:, 2, :, :]
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :] + 2.03211 * image_yuv[:, 1, :, :]


class JpegCompression(nn.Module):
    def __init__(self, device, yuv_keep_weights=(25, 9, 9)):
        super(JpegCompression, self).__init__()
        self.device = device

        self.dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).unsqueeze(1).to(device)
        self.idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).unsqueeze(1).to(device)
        self.yuv_keep_weights = yuv_keep_weights
        self.jpeg_mask = None
        self.create_mask((1024, 1024))

    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weights):
                mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):

        if len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple of length 3.")

        mask = self.jpeg_mask[:image_shape[0], :image_shape[1], :image_shape[2]]

        return mask.clone()

    def apply_conv(self, image, filter_type: str):
        filters = self.dct_conv_weights if filter_type == 'dct' else self.idct_conv_weights
        image = image.to(self.device)  # 确保输入图像也在正确的设备上
        image_conv_channels = []
        for channel in range(image.shape[1]):
            image_yuv_ch = image[:, channel, :, :].unsqueeze(1)
            image_conv = F.conv2d(image_yuv_ch, filters, stride=8)
            image_conv = image_conv.permute(0, 2, 3, 1).view(image_conv.shape[0], image_conv.shape[3],
                                                             image_conv.shape[2], 8, 8).permute(0, 1, 3, 2,
                                                                                                4).contiguous()
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1] * image_conv.shape[2],
                                         image_conv.shape[3] * image_conv.shape[4]).unsqueeze(1)
            image_conv_channels.append(image_conv)
        return torch.cat(image_conv_channels, dim=1)

    def forward(self, steg):

        noised_image = steg
        pad_height = (8 - noised_image.shape[2] % 8) % 8
        pad_width = (8 - noised_image.shape[3] % 8) % 8

        noised_image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(noised_image)
        image_yuv = torch.empty_like(noised_image)
        rgb2yuv(noised_image, image_yuv)
        image_dct = self.apply_conv(image_yuv, 'dct')
        mask = self.get_mask(image_dct.shape[1:])
        image_dct_mask = torch.mul(image_dct, mask)

        image_idct = self.apply_conv(image_dct_mask, 'idct')
        image_ret_padded = torch.empty_like(image_dct)
        yuv2rgb(image_idct, image_ret_padded)

        noised_steg = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                              :image_ret_padded.shape[3] - pad_width].clone()

        return noised_steg


