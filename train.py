import torch
import logging
from PIL import Image
import config
import utils
import numpy as np
import Loss
from model import *
from network.waveletTrans import DWT,IWT
from network.LowPassfitter import LowpassFilter
from noise.guassian import GaussianNoise
from noise.jpeg_compression import JpegCompression
from noise.noiser import Noiser
from noise.cropout import Cropout
from noise.rotate import RotateImage
# from noise.crop import Crop
from noise.dropout import Dropout
from tqdm import tqdm
from metrics import Metrics


def train():
    # 日志记录器的初始化
    logger_name = "train_log"
    log_root = "log/"
    logger = utils.log_starter(logger_name, log_root, level=logging.INFO, out='tofile')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(8).to(device)
    utils.init_model(model, device=device)

    utils.model_structure(model,logger)
    model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optim = torch.optim.Adam(params_trainable, lr=config.lr, betas=config.betas, eps=1e-8, weight_decay=config.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, config.weight_step, gamma=config.gamma)
    dwt = DWT()
    iwt = IWT()
    train_data = utils.get_dataloader(type='train')
    val_data = utils.get_dataloader(type='val')
    val_epoch=0
    lpf = LowpassFilter(kernel_size=5)
    psnr_values = []
    ssim_values = []
    for epoch in range(config.epoch):
        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []
        h_loss_history = []
        #train
        for data,_ in tqdm(train_data,desc=f"train_epoch_{epoch+1}", leave=True, unit='it'):
            '''initialization'''
            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            img_input = torch.cat((cover_input, secret_input), 1)
            # print(f'img_input:{img_input.max()}')
            '''encoder'''
            output = model(img_input)
            # print(f'Encoder:{output.max()}')
            output_steg = output.narrow(1, 0, 4 * config.channels_in)
            steg = iwt(output_steg, device)
            # steg = lpf(steg)
            # output_z = output.narrow(1, 4 * config.channels_in, output.shape[1] - 4 * config.channels_in)
            # output_z = utils.gauss_noise(output_z.shape, device)
            '''noise layer'''
            if epoch % 3 == 0:
                pass
            else:
                noise = Noiser(mode = 'none')
                noise.add_noise_layer(layer=GaussianNoise())
                noise.add_noise_layer(layer=JpegCompression(device))
                noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.4, 0.6)))
                # noise.add_noise_layer(layer=Cropout(0.3,0.7))
                # noise.add_noise_layer(layer=Resize((0.5,1.5)))
                img = noise(steg)

                output_steg = dwt(img)
            '''decoder'''
            output_steg = output_steg.to(device)
            random_init = torch.randn_like(output_steg,dtype=torch.float32).to(device)
            output_rev = torch.cat((output_steg, random_init), 1)
            output_image = model(output_rev, rev=True)
            # print(f'Decoder:{output_image.max()}')
            secret_rev = output_image.narrow(1, 4 * config.channels_in, output_image.shape[1] - 4 * config.channels_in)
            secret_rev = iwt(secret_rev, device)
            '''calculate loss'''
            # print(steg)
            g_loss = Loss.guide_loss(steg, cover, device=device)
            r_loss = Loss.reconstruction_loss(secret_rev, secret, device=device)
            h_loss = Loss.histogram_loss(steg, cover, device=device)
            steg_low = output_steg.narrow(1, 0, config.channels_in)
            cover_low = cover_input.narrow(1, 0, config.channels_in)
            l_loss = Loss.low_frequency_loss(steg_low, cover_low, device=device)

            # total_loss = config.lamda_reconstruction * r_loss + config.lamda_guide * g_loss + config.lamda_low_frequency * l_loss
            total_loss = config.lamda_guide * g_loss + config.lamda_reconstruction * r_loss
            # total_loss = config.lamda_guide * g_loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e7)
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            g_loss_history.append([g_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            h_loss_history.append([h_loss.item(), 0.])

        epoch_loss = np.mean(np.array(loss_history),axis=0)
        g_loss = np.mean(np.array(g_loss_history),axis=0)
        l_loss = np.mean(np.array(l_loss_history),axis=0)
        r_loss = np.mean(np.array(r_loss_history),axis=0)
        h_loss = np.mean(np.array(h_loss_history),axis=0)
        current_lr = optim.param_groups[0]['lr']
        logger.info(f"Epoch: {epoch+1}/{config.epoch}, Current lr: {current_lr}, Loss: {epoch_loss[0].item():3f},"
                    f"g_loss: {g_loss[0].item():3f}|r_loss: {r_loss[0].item():3f}|l_loss: {l_loss[0].item():3f}")
        print(f"Current lr: {current_lr}  |Loss: {epoch_loss[0].item():3f}|g_loss: {g_loss[0].item():3f}|"
              f"r_loss: {r_loss[0].item():3}|l_loss: {l_loss[0].item():3}|h_loss：{h_loss[0].item():3f}")

        #val
        if (epoch+1) % config.val_freq == 0:
            val_epoch += 1
            PSNR, SSIM, BER = [], [], []
            model.eval()
            # print(f'Val Epoch:{val_epoch}')
            with torch.no_grad():
                for idx, (data, target) in tqdm(enumerate(val_data),desc=f"val_epoch_{val_epoch}",leave=True, unit='it'):
                    # id = val_epoch
                    data = data.to(device)
                    cover = data[data.shape[0] // 2:]
                    secret = data[:data.shape[0] // 2]
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)
                    img_input = torch.cat((cover_input, secret_input), 1)
                    '''encoder'''
                    output = model(img_input)
                    output_steg = output.narrow(1, 0, 4 * config.channels_in)
                    steg = iwt(output_steg, device)
                    output_z = output.narrow(1, 4 * config.channels_in, output.shape[1] - 4 * config.channels_in)
                    output_z = utils.gauss_noise(output_z.shape, device)

                    '''noise layer'''
                    # noise = Noiser(mode='random',layer_num=2)
                    # noise.add_noise_layer(layer=GaussianNoise())
                    # noise.add_noise_layer(layer=JpegCompression(device))
                    # noise.add_noise_layer(layer=Cropout(0.3, 0.7))
                    # noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.4, 0.6)))
                    # noise.add_noise_layer(layer=Resize((0.5, 1.5)))
                    # img = noise(steg)
                    # output_steg = dwt(img)
                    '''decoder'''
                    output_steg = output_steg.to(device)
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = model(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * config.channels_in,
                                                     output_image.shape[1] - 4 * config.channels_in)
                    secret_rev = iwt(secret_rev, device)

                    '''save images'''
                    utils.save_images('result/val_images', cover, secret, steg, secret_rev,epoch = val_epoch,id = idx)

                    '''calculate metrics'''
                    metrics = Metrics(Image.open(f'result/val_images/secret/secret_{val_epoch}_{idx:03d}.png'),
                                      Image.open(f'result/val_images/secret_rev/secret_rev_{val_epoch}_{idx:03d}.png'))
                    psnr = metrics.psnr()
                    PSNR.append(psnr)
                    ssim = metrics.ssim()
                    SSIM.append(ssim)
                    ber = metrics.ber()
                    BER.append(ber)

            logger.info(f"Avg PSNR:{np.mean(PSNR)},Avg SSIM:{np.mean(SSIM)},Avg BER:{np.mean(BER)}")
            print(f"Avg PSNR:{np.mean(PSNR)},Avg SSIM:{np.mean(SSIM)},Avg BER:{np.mean(BER)}")
            torch.save({'opt': optim.state_dict(), 'net': model.state_dict()},
                       config.MODEL_PATH + f'model_{epoch+1}_val.pt')
        if len(psnr_values and ssim_values) > 0:
            avg_psnr = sum(psnr_values) / len(psnr_values)
            avg_ssim = sum(ssim_values) / len(ssim_values)
            psnr_values.append(avg_psnr)
            ssim_values.append(avg_ssim)
            print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")
            logger.info(f"Epoch {epoch + 1}, Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")
        else:
            avg_ssim = None

        weight_scheduler.step()
    torch.save({'opt':optim.state_dict(),'net':model.state_dict()},config.MODEL_PATH+f'model_{config.epoch}.pt')
    utils.plot_metrics(psnr_values, ssim_values,f"{config.METRIC_PATH}/metrics_over_epochs.png")

if __name__ == '__main__':
    train()