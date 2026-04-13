import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
from PIL import Image
import numpy as np
from .augmenter import Augmenter
import numbers
import mxnet as mx
import pandas as pd
import cv2

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
    


class BaseMXDataset(Dataset):
    def __init__(self, root_dir, swap_color_channel=False):
        super(BaseMXDataset, self).__init__()
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        path_imglst = os.path.join(root_dir, 'train.lst')

        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # grad image index from the record and know how many images there are.
        # image index could be occasionally random order. like [4,3,1,2,0]
        s = self.record.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))
        print('record file length', len(self.imgidx))

        record_info = []
        for idx in self.imgidx:
            s = self.record.read_idx(idx)
            header, _ = mx.recordio.unpack(s)
            label = header.label
            row = {'idx': idx, 'path': '{}/name.jpg'.format(label), 'label': label}
            record_info.append(row)
        self.record_info = pd.DataFrame(record_info)

        self.swap_color_channel = swap_color_channel
        if self.swap_color_channel:
            print('[INFO] Train data in swap_color_channel')

    def read_sample(self, index):
        idx = self.imgidx[index]
        s = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        return sample, label

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.imgidx)



class AugmentRecordDataset(BaseMXDataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 output_dir='./'
                 ):
        super(AugmentRecordDataset, self).__init__(root_dir,
                                                   swap_color_channel=swap_color_channel,
                                                   )
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.transform = transform
        self.output_dir = output_dir

    def __getitem__(self, index):
        sample, target = self.read_sample(index)

        sample = self.augmenter.augment(sample)
        sample_save_path = os.path.join(self.output_dir, 'training_samples', 'sample.jpg')
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

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