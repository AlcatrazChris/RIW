import torch
from imageworks.color_loss import LabLoss
import torch.nn as nn
import torch.nn.functional as F
from imageworks import rgb2ycbcr,texture


def guide_loss(output, bicubic_image,device):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # print(f"output range{output.max()-output.min()}, bicubic range{bicubic_image.max()-bicubic_image.min()}")
    # print(f"output size{output.shape}  bicubic size{bicubic_image.shape}")
    loss = loss_fn(output, bicubic_image)
    # print(loss.item())
    return loss.to(device)


def reconstruction_loss(rev_input, input, device):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input, device):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)

def histogram_loss(hist_input, gt_input, device):
    hist_loss = LabLoss()
    loss = hist_loss(hist_input, gt_input)
    return loss.to(device)

def denoider_loss(original, denoised, device):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(original, denoised)
    return loss.to(device)


def normalize_cr(cr_channel):
    cr_min = cr_channel.min()
    cr_max = cr_channel.max()
    return (cr_channel - cr_min) / (cr_max - cr_min)


def color_loss(cover,steg,device):
    loss_fn = rgb2ycbcr.ColorWeightedMSELoss(g_function=normalize_cr)
    loss = loss_fn(cover,steg)
    return loss.to(device)

def texture_loss(cover,steg,device):
    loss_fn = texture.TextureLoss()
    loss = loss_fn(cover,steg)
    return loss.to(device)