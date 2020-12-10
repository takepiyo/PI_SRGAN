from os import listdir
from os.path import join

from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn as nn
import torch

import pickle
import numpy as np

import gc


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def make_dataset_from_pickle(dataset_file, upscale_factor, out_dir, split_rate=0.9):
    with open(dataset_file, 'rb') as f:
        data_dict = pickle.load(f)
    u_v_p = np.stack([
        data_dict['u'].transpose(2, 1, 0),
        data_dict['v'].transpose(2, 1, 0),
        data_dict['p'].transpose(2, 1, 0)], axis=3)[:2000, :, :, :]

    np.random.shuffle(u_v_p)
    u_v_p_train, u_v_p_valid = np.split(
        u_v_p, [int(split_rate * u_v_p.shape[0])], 0)

    del data_dict
    gc.collect()

    train_dataset = DatasetFromPickle(u_v_p_train, upscale_factor)
    valid_dataset = DatasetFromPickle(u_v_p_valid, upscale_factor)

    with open(out_dir + "/train.pickle", 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(out_dir + "/valid.pickle", 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset, valid_dataset


class DatasetFromPickle(Dataset):
    def __init__(self, data, upscale_factor):
        super(DatasetFromPickle, self).__init__()
        data = data.astype(np.float32)
        self.data = data
        self.number, self.crop_size, _, _ = data.shape
        self.upscale_factor = upscale_factor

        # self.hr_transform = Compose([ToTensor()
        #                              ])
        # self.lr_transform = Compose([ToPILImage(),
        #                              Resize(self.crop_size // upscale_factor,
        #                                     interpolation=Image.BICUBIC),
        #                              ToTensor()])

        self.restore_transform = Compose([ToPILImage(),
                                          Resize(
                                              self.crop_size, interpolation=Image.NEAREST),
                                          ToTensor()])

        self.low_pass_filter_conv = self.get_low_pass_filter()

    def __getitem__(self, index):
        normalized = self.normalize_space(self.data[index, :, :])
        # hr_image = self.hr_transform(normalized)
        hr_image = ToTensor()(normalized)
        # lr_image = self.lr_transform(hr_image)
        lr_image = self.low_pass_filter_conv(hr_image.unsqueeze(0)).squeeze(0)
        restored_image = self.restore_transform(lr_image)
        return lr_image, restored_image, hr_image

    def __len__(self):
        return self.number

    def normalize_space(self, data):
        #  data(width, height, channel(u v p))
        vars = np.var(data, axis=(0, 1))
        vars = np.sqrt(np.sum(vars))
        data[:, :, 0] = data[:, :, 0] / vars
        data[:, :, 1] = data[:, :, 1] / vars
        data[:, :, 2] = data[:, :, 2] / (vars ** 2)
        return data

    # only 4 upscale factor is adopted
    def get_low_pass_filter(self):
        filter = nn.Conv2d(3, 3, 7, 4, 3, groups=3, bias=False)
        w_0 = 0.22723004 * 2
        w_1 = 0.20002636
        w_2 = 0.13638498
        w_3 = 0.04997364

        one_channel_weight = torch.tensor([[0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
                                           [w_3, w_2, w_1, w_0, w_1, w_2, w_3],
                                           [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0]], dtype=torch.float32)

        filter.weight[0, 0, :, :] = one_channel_weight
        filter.weight[1, 0, :, :] = one_channel_weight
        filter.weight[2, 0, :, :] = one_channel_weight

        return filter


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor,
                          interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x)
                             for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x)
                             for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize(
            (self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


if __name__ == '__main__':
    filter = nn.Conv2d(3, 3, 7, 4, 3, groups=3, bias=False)
    # nn.init.xavier_uniform_(filter.weight)

    w_0 = 0.22723004 * 2
    w_1 = 0.20002636
    w_2 = 0.13638498
    w_3 = 0.04997364

    one_channel_weight = torch.tensor([[0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
                                       [w_3, w_2, w_1, w_0, w_1, w_2, w_3],
                                       [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0]], dtype=torch.float32)

    filter.weight[0, 0, :, :] = one_channel_weight
    filter.weight[1, 0, :, :] = one_channel_weight
    filter.weight[2, 0, :, :] = one_channel_weight

    img = torch.ones([5, 3, 128, 128])
    print(filter.weight)

    output = filter(img)

    a = 0

    for one_img in output:
        print('=====================')
        print(one_img)
