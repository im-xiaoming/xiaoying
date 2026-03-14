import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image
import numpy as np
from .augmenter import Augmenter

class CustomImageFolderDataset(datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=transforms.ToTensor(),
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.augmenter = Augmenter(crop_augmentation_prob,
                                   photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        if self.swap_color_channel:
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        sample = self.augmenter.augment(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
    
    
    
class FiveValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):

        self.dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cfp_ff": 3}

        self.val_data_dict = val_data_dict
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = []
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
        assert key_orders == ['agedb_30', 'cfp_fp', 'lfw', 'cfp_ff']

        try:
          self.all_imgs = read_memmap(concat_mem_file_name)
        except:
          self.all_imgs = np.concatenate(all_imgs)
          make_memmap(concat_mem_file_name, self.all_imgs)
          self.all_imgs = read_memmap(concat_mem_file_name)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame)
        assert len(self.all_issame) == len(self.all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)


def val_dataset(data_root='data', val_data_path=None, concat_mem_file_name='mm.dat'):
    val_data = get_val_data(data_root)
    agedb_30, agedb_30_issame, cfp_fp, cfp_fp_issame, lfw, lfw_issame, cfp_ff, cfp_ff_issame = val_data
    val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cfp_ff": (cfp_ff, cfp_ff_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict,
                                        os.path.join(data_root, concat_mem_file_name))

    return val_dataset


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
    return agedb_30, agedb_30_issame, cfp_fp, cfp_fp_issame, lfw, lfw_issame, cfp_ff, cfp_ff_issame

def get_val_pair(path, name):
    mem_file_name = os.path.join(path, name + '.dat')
    print('loading validation data memfile')
    np_array = read_memmap(mem_file_name)

    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return np_array, issame


def make_memmap(mem_file_name, np_to_copy):
    memmap_configs = dict()
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape)
    memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)
    json.dump(memmap_configs, open(mem_file_name + '.conf', 'w'))
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = np_to_copy[:]
    mm.flush()
    return mm


def read_memmap(mem_file_name):
    with open(mem_file_name + '.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', \
                         shape=tuple(memmap_configs['shape']), \
                         dtype=memmap_configs['dtype'])