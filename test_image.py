import argparse
import time
import pickle

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.utils as utils
from data_utils import TrainDatasetFromFolder, make_dataset_from_pickle
from torch.utils.data import DataLoader

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

# crop_size = dataset.crop_size
crop_size = 128

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

dataloader = DataLoader(dataset, batch_size=1, shuffle=False).__iter__()

if INDEX != 0:
  for i in range(INDEX - 1):
    dataloader.__next__()

lr, hr_restore, hr = dataloader.__next__()

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
# utils.save_image(lr.data.cpu().squeeze(0), os.path.join(
#     image_out_dir, '{}_{}_lr.png'.format(PICKLE_TYPE, INDEX)))



