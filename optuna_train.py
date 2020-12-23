import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, make_dataset_from_pickle
from loss import GeneratorLoss
from model import Generator, Discriminator

import matplotlib.pyplot as plt

def main(weight_tuple, pi_weight_tuple, dataset_tuple):

    # CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = 4
    NUM_EPOCHS = 100
    #OUT_DIR = "model/" + opt.output_dir
    #SAVE_PER_EPOCH = opt.save_per_epoch
    BATCH_SIZE = 100

    train_set, val_set = dataset_tuple
    
    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel()
                                         for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel()
                                             for param in netD.parameters()))

    # loss_weight = (1.0, 0.001, 0.006, 2e-8, 0.5)
    # lambda_params = (0.5, 0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator_criterion = GeneratorLoss(
        weight_tuple, train_set.get_params(), pi_weight_tuple, device)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}
    eval_mse_error_list = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        torch.autograd.set_detect_anomaly(True)

        for data, _, target, _ in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            # print(data[0].size())
            # fig = plt.figure()
            # plt.imshow(data[0].detach().cpu().numpy().transpose(1,2,0))
            # fig.savefig("out_srf_4_data/large_cylinder/train_LR/a.jpg")

            # plt.imshow(target[0].detach().cpu().numpy().transpose(1,2,0))
            # fig.savefig("out_srf_4_data/large_cylinder/train_LR/HR.jpg")
            # exit()

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] /
                running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0,
                              'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr, _ in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (
                    valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / \
                    valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
            eval_mse_error_list.append(valing_results['mse'])

    return min(eval_mse_error_list)
            #     val_images.extend(
            #         [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
            #          display_transform()(sr.data.cpu().squeeze(0))])
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            # index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(
            #         image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1

