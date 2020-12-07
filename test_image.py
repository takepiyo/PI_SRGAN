import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from data_utils import TrainDatasetFromFolder

import os
import matplotlib.pyplot as plt
# %matplotlib inline

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str,
                    choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str,
                    help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth',
                    type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = 8
TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = "data/large_cylinder/HR/000286.jpg"
# MODEL_NAME = "netG_epoch_8_55.pth"
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

crop_size = 512

model = Generator(UPSCALE_FACTOR).eval()
criterion = torch.nn.MSELoss().cuda()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load(
        'epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

#image = Image.open(IMAGE_NAME)

#data = TrainDatasetFromFile(IMAGE_NAME, crop_size, UPSCALE_FACTOR)
data = TrainDatasetFromFolder("my_data", crop_size, UPSCALE_FACTOR)
IMAGE_NAME = os.path.basename(IMAGE_NAME)
lr_image, hr_image = data[0]
LR_img, HR_img = ToPILImage()(lr_image.data.cpu()), ToPILImage()(hr_image.data.cpu())
LR_img.save('my_result/LR_' + IMAGE_NAME)
HR_img.save('my_result/HR_' + IMAGE_NAME)

lr_image = lr_image.unsqueeze(0)

# LR_img = ToPILImage()(lr_image[0].data.cpu())
# LR_img.save('my_result/LR_' + IMAGE_NAME)

if TEST_MODE:
    lr_image = lr_image.cuda()

# start = time.clock()
out_image = model(lr_image)
# elapsed = (time.clock() - start)
#print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out_image[0].data.cpu())
out_img.save('my_result/PRED_' + IMAGE_NAME)
#out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
if TEST_MODE:
    print("loss", criterion(hr_image.view(-1).cuda(), out_image.view(-1)).item())
else:
    print("loss", criterion(hr_image.view(-1), out_image.view(-1)).item())
