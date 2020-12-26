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

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=128, type=int,
#                     help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100,
                    type=int, help='train epoch number')
parser.add_argument('--output_dir',
                    type=str, help='output directory in model directory')
parser.add_argument('--save_per_epoch', default=10,
                    type=int, help='save per epoch number')
parser.add_argument('--batch_size', default=1,
                    type=int, help='batch_size')
parser.add_argument('--number_of_data', default=5000,
                    type=int, help='the number of data')
parser.add_argument('--pickle',
                    type=str, help='pickle file path')
if __name__ == '__main__':
    opt = parser.parse_args()

    # CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    OUT_DIR = "model/" + opt.output_dir
    SAVE_PER_EPOCH = opt.save_per_epoch
    BATCH_SIZE = opt.batch_size
    DATA_LENGTH = opt.number_of_data
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # train_set = TrainDatasetFromFolder(
    #     '/content/drive/My Drive/SRGAN/data/large_cylinder/HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder(
    #     '/content/drive/My Drive/SRGAN/data/large_cylinder/HR', upscale_factor=UPSCALE_FACTOR)

    train_set, val_set = make_dataset_from_pickle(
        opt.pickle, UPSCALE_FACTOR, OUT_DIR, DATA_LENGTH)

    train_loader = DataLoader(
        dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR, 64)
    print('# generator parameters:', sum(param.numel()
                                         for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel()
                                             for param in netD.parameters()))

    # 赤と青が出た重み
    #loss_weight = (10.0, 0.001, 0.006, 2e-8, 0.001) 
    #image_loss_weight = (0.3, 0.4, 0.3)
    #lambda_params = (0.4, 0.001)
    #image loss の max正規化なしで緑と赤が出たやつ
    # loss_weight = (10.0, 0.001, 0.006, 2e-8, 0.001) 
    # image_loss_weight = (0.0925, 0.9, 0.0075)
    # lambda_params = (0.4, 0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator_criterion = GeneratorLoss(
        loss_weight, image_loss_weight, train_set.get_params(), lambda_params, device)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           'g_loss': 0, 'd_score': 0, 'g_score': 0}
        detailed_loss = {'adversarial': 0, 'perception': 0, 
                         'image_loss': 0, 'tv_loss': 0, 'pi_loss': 0}

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

            # print(z)

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
            
            sum(g_loss).backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += sum(g_loss).item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            detailed_loss['adversarial'] += g_loss[1].item() * batch_size
            detailed_loss['perception'] += g_loss[2].item() * batch_size
            detailed_loss['image_loss'] += g_loss[0].item() * batch_size
            detailed_loss['tv_loss'] += g_loss[3].item() * batch_size
            detailed_loss['pi_loss'] += g_loss[4].item() * batch_size


            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f image_loss: %.4f pi_loss: %.4f percep: %.4f tv: %.4f adver: %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] /
                running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                detailed_loss['image_loss'] / running_results['batch_sizes'],
                detailed_loss['pi_loss'] / running_results['batch_sizes'], 
                detailed_loss['perception'] / running_results['batch_sizes'], 
                detailed_loss['tv_loss'] / running_results['batch_sizes'],
                detailed_loss['adversarial'] / running_results['batch_sizes']))

        netG.eval()
        out_path = os.path.join(OUT_DIR,'training_results/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0,
                              'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            # for val_lr, val_hr_restore, val_hr, lr_expanded in val_bar:
            for val_lr, val_hr_restore, val_hr, lr_expanded in val_bar:
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

                val_images.extend(
                    [lr_expanded.squeeze(0), hr.data.cpu().squeeze(0),
                     sr.data.cpu().squeeze(0)])
            val_images = torch.stack(val_images)
            #val_images = torch.chunk(val_images, val_images.size(0) // 15)
            #val_save_bar = tqdm(val_images, desc='[saving training results]')
            image = torch.chunk(val_images, val_images.size(0) // 15)[0]
            #val_save_bar = tqdm(val_images, desc='[saving training results]')
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + 'epoch_%d.png' % (epoch), padding=5)

            #index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(
            #         image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1

        # save model parameters
        if epoch % SAVE_PER_EPOCH == 0:
            torch.save(netG.state_dict(), os.path.join(OUT_DIR, 'netG_epoch_%d_%d.pth' %
                                                       (UPSCALE_FACTOR, epoch)))
            torch.save(netD.state_dict(), os.path.join(OUT_DIR, 'netD_epoch_%d_%d.pth' %
                                                       (UPSCALE_FACTOR, epoch)))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(
            running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(
            running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(
            running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(
            running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) +
                              '_train_results.csv', index_label='Epoch')
