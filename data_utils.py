from os import listdir
from os.path import join

from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

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

    with open(out_dir + "/train.pickle", 'rb') as f:
        pickle.dump(train_dataset, f)
    with open(out_dir + "valid.pickle", 'rb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset, valid_dataset


class DatasetFromPickle(Dataset):
    def __init__(self, data, upscale_factor):
        super(DatasetFromPickle, self).__init__()
        data = data.astype(np.float32)
        self.data = data
        self.number, crop_size, _, _ = data.shape
        self.upscale_factor = upscale_factor

        self.hr_transform = Compose([ToTensor()
                                     ])
        self.lr_transform = Compose([ToPILImage(),
                                     Resize(crop_size // upscale_factor,
                                            interpolation=Image.BICUBIC),
                                     ToTensor()])
        self.restore_transform = Compose([ToPILImage(),
                                          Resize(
                                              crop_size, interpolation=Image.BICUBIC),
                                          ToTensor()])

    def __getitem__(self, index):
        hr_image = self.hr_transform(self.data[index, :, :])
        lr_image = self.lr_transform(hr_image)
        restored_image = self.restore_transform(lr_image)
        return lr_image, restored_image, hr_image

    def __len__(self):
        return self.number


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
