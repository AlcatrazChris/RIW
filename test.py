import os
import numpy as np
import torch
import torchvision
import config
import utils
import logging
from PIL import Image
from metrics import Metrics
from model import Model
from network.waveletTrans import DWT,IWT
from network.LowPassfitter import LowpassFilter
from noise.noiser import Noiser
from noise.jpeg_compression import JpegCompression
from noise.guassian import GaussianNoise
from noise.dropout import Dropout
# from noise.crop import Crop
from noise.cropout import Cropout


def load(name):
    state_dicts = torch.load(name, map_location=device)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def save_images(images, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for img, label in zip(images, labels):
        torchvision.utils.save_image(img, os.path.join(directory, label))

def calculate_metrics(pred, target, logger, id):
    metrics = Metrics(pred, target)
    psnr = metrics.psnr()
    ssim = metrics.ssim()
    ber = metrics.ber()
    logger.info(f"id:{id}: PSNR: {psnr:.4f}| SSIM: {ssim:.4f}| BER: {ber:.4f}")
    return psnr, ssim, ber

#setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = utils.log_starter("test_log", "runs/log/", level=logging.INFO, out='tofile')
model = Model(8).to(device)
utils.init_model(model, device=device)
params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
optim = torch.optim.Adam(params_trainable, lr=config.lr, betas=config.betas, eps=1e-6, weight_decay=config.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, config.weight_step, gamma=config.gamma)

utils.load_from_hinet('model/model_580_val.pt',model=model, device=device,optim=optim)
# load('model/best10_0517.pt')
# load('model/model_290_val.pt')
model.eval()
dwt, iwt = DWT(), IWT()
test_data = utils.get_dataloader(type='test')
lpf = LowpassFilter(kernel_size=3)
PSNR_C, SSIM_C, BER_C = [], [], []
PSNR_S, SSIM_S, BER_S = [], [], []
def test():
    with torch.no_grad():
        for id, (data, target) in enumerate(test_data):
            '''Data initial'''
            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            img_input = torch.cat((cover_input, secret_input), 1)
            '''Encoder'''
            output = model(img_input)
            output_steg = output.narrow(1, 0, 4 * config.channels_in)
            steg = iwt(output_steg, device)
            print(steg.shape)
            steg = lpf(steg)
            output_z = output.narrow(1, 4 * config.channels_in, output.shape[1] - 4 * config.channels_in)
            output_z = utils.gauss_noise(output_z.shape, device)
            '''Noise layer'''
            noise = Noiser()
            # noise.add_noise_layer(layer=GaussianNoise(mean=0.,std=0.05))
            # noise.add_noise_layer(layer=JpegCompression(device))
            # noise.add_noise_layer(layer=Cropout())
            # noise.add_noise_layer(RotateImage(30))
            # noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.4, 0.6)))
            # noise.add_noise_layer(layer=Resize((0.5, 0.7)))
            img = noise(steg)
            output_steg = dwt(img)
            '''Decoder'''
            output_steg = output_steg.to(device)
            output_rev = torch.cat((output_steg, output_z), 1)
            output_image = model(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * config.channels_in,
                                             output_image.shape[1] - 4 * config.channels_in)
            secret_rev = iwt(secret_rev, device)
            resi_cover = (steg - cover) * 20
            resi_secret = (secret_rev - secret) * 20

            '''save images'''
            utils.save_images(config.IMAGE_PATH,cover, secret, steg, secret_rev, id=id)

            '''calculate metrics'''
            metrics_s = Metrics(Image.open(config.IMAGE_PATH+'/secret/secret_' + '%.3d.png' % id), Image.open(config.IMAGE_PATH+'/secret_rev/secret_rev_' + '%.3d.png' % id))
            metrics_c = Metrics(Image.open(config.IMAGE_PATH+'/cover/cover_'+ '%.3d.png' % id),Image.open(config.IMAGE_PATH+'/steg/steg_' + '%.3d.png' % id))
            psnr_c = metrics_c.psnr()
            psnr_s = metrics_s.psnr()
            PSNR_C.append(psnr_c)
            PSNR_S.append(psnr_s)
            ssim_c = metrics_c.ssim()
            ssim_s = metrics_s.ssim()
            SSIM_C.append(ssim_c)
            SSIM_S.append(ssim_s)
            # ber = metrics.ber()
            # BER.append(ber)
            logger.info(f"id:{id}: PSNR_C: {psnr_c:.4f}| SSIM_C: {ssim_c:4f}| PSNR_S:{psnr_s:4f}| SSIM_S:{ssim_s}")
        logger.info(f"Avg PSNR_C:{np.mean(PSNR_C)},Avg SSIM_C:{np.mean(SSIM_C)},Avg PSNR_S:{np.mean(PSNR_S)},Avg SSIM_S:{np.mean(SSIM_S)}")

test()