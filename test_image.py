import argparse
import time
import pickle

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.utils as utils
from data_utils import TrainDatasetFromFolder, make_dataset_from_pickle

import os
import matplotlib.pyplot as plt
# %matplotlib inline

from model import Generator


parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--test_mode', default='GPU', type=str,
#                     choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--dataset_type', default='valid', type=str,
                    choices=['train', 'valid'], help='choice dataset type')
parser.add_argument('--model_name', type=str,
                    help='generator model relative path')
parser.add_argument('--data_index', type=int,
                    help='dataset index')
opt = parser.parse_args()

TEST_MODE = True if torch.cuda.is_available() else False
PICKLE_TYPE = opt.dataset_type
MODEL_NAME = opt.model_name
INDEX = opt.data_index

model_dir = os.path.dirname(MODEL_NAME)
image_out_dir = os.path.join(model_dir, 'result')

if not os.path.exists(image_out_dir):
    os.makedirs(image_out_dir)

with open(os.path.join(model_dir, PICKLE_TYPE + '.pickle'), 'rb') as f:
    dataset = pickle.load(f)

crop_size = dataset.crop_size
UPSCALE_FACTOR = dataset.upscale_factor

model = Generator(UPSCALE_FACTOR).eval()

if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
    criterion = torch.nn.MSELoss().cuda()
else:
    model.load_state_dict(torch.load(
        MODEL_NAME, map_location=lambda storage, loc: storage))
    criterion = torch.nn.MSELoss()

# data = TrainDatasetFromFolder("my_data", crop_size, UPSCALE_FACTOR)

lr, hr_restore, hr = dataset[INDEX]

if torch.cuda.is_available():
    lr = lr.cuda()
    hr = hr.cuda()
sr = model(lr)

print("loss", criterion(hr.view(-1), sr.view(-1)).item())

images = torch.stack(
    [hr_restore.squeeze(0), hr.data.cpu().squeeze(0), sr.data.cpu().squeeze(0)])

images = utils.make_grid(images, nrow=3, padding=5)
utils.save_image(images, os.path.join(
    image_out_dir, '{}_{}.png'.format(PICKLE_TYPE, INDEX)))

# LR_img, HR_img = ToPILImage()(lr_image.data.cpu()), ToPILImage()(hr_image.data.cpu())
# LR_img.save('my_result/LR_' + IMAGE_NAME)
# HR_img.save('my_result/HR_' + IMAGE_NAME)

# lr_image = lr_image.unsqueeze(0)

# LR_img = ToPILImage()(lr_image[0].data.cpu())
# LR_img.save('my_result/LR_' + IMAGE_NAME)

# if TEST_MODE:
#     lr_image = lr_image.cuda()

# start = time.clock()
# out_image = model(lr_image)
# elapsed = (time.clock() - start)
# print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out_image[0].data.cpu())
# out_img.save('my_result/PRED_' + IMAGE_NAME)
# out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
# if TEST_MODE:
#     print("loss", criterion(hr_image.view(-1).cuda(), out_image.view(-1)).item())
# else:
#     print("loss", criterion(hr_image.view(-1), out_image.view(-1)).item())
