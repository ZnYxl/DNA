"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from collections import defaultdict
from utils.utils import is_None, getmaxlen, group_shuffle

Separator='==============================='



import os
import pathlib
from torch.utils.data import Dataset

import os
import pathlib
from torch.utils.data import Dataset


class MyDataset(Dataset):
    Separator = '---'

    def __init__(self, path_dict: dict, datasets: list = None, mode: str = 'train'):
        assert mode in ['train',  'val', 'test'], "mode must be 'train' or 'test'"

        self.path_dict = path_dict
        self.datasets = datasets if datasets else list(path_dict.keys())
        self.mode = mode
        self.x_list, self.y_data = self.load_data_wrapper()

        assert len(self.x_list) == len(self.y_data), "x_list and y_data must have the same length"

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.x_list):
            raise IndexError(f"Index {index} out of range for x_list of length {len(self.x_list)}")
        X = self.x_list[index]
        Y = self.y_data[index]
        return X, Y


    def load_data_wrapper(self):
        x_data = []
        y_data = []
        x_list = []

        for dataset in self.datasets:
            source_reads_path = self.path_dict[dataset] / self.mode / 'reads.txt'
            source_reference_path = self.path_dict[dataset] / self.mode / 'reference.txt'

            try:
                with open(source_reads_path, 'r') as f1, open(source_reference_path, 'r') as f2:
                    f1_r = f1.readlines()
                    f2_r = f2.readlines()
                    id_list = []
                    id = 0

                    for x_line in f1_r:
                        x_line = x_line.strip('\n')
                        if x_line != Separator:
                            x_data.append(''.join(x_line))
                        elif x_line == Separator and x_data != []:
                            x_list.append(x_data)
                            x_data = []
                            id += 1
                        elif x_line == Separator and x_data == []:
                            id_list.append(id)
                            id += 1
                    for j, y_line in enumerate(f2_r):
                        if j + 1 not in id_list:
                            y_line = y_line.strip('\n')
                            y_data.append(''.join(y_line))
            except FileNotFoundError:
                print(f"Warning: {source_reads_path} or {source_reference_path} not found.")

        return x_list, y_data


class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        nums = []
        for i in range(len(self.data)):
            num = len(self.data[i][0])
            nums.append(num)
        indices = group_shuffle(nums)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class CustomBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (i < len(sampler_list) - 1 and
                    len(self.sampler.data[idx][0]) !=
                    len(self.sampler.data[sampler_list[i + 1]][0])):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
            i += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class collater():
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def __call__(self, batch):
        enc=OneHotEncoder(categories = [['A', 'C', 'G', 'T']])
        fea_batch=[]
        label_batch=[]

        for i in range(len(batch)):
            dic=batch[i]
            fea_batch.append(dic[0])
            label_batch.append(dic[1])

        max_len=self.maxlen

        dim1 = []
        dim2 = []
        for i in range(len(fea_batch)):
            for j in range(len(fea_batch[0])):
                fea_arr = np.array(list(fea_batch[i][j])).reshape(-1, 1)
                fea_onehot = enc.fit_transform(fea_arr).toarray()
                fea_onehot = torch.tensor(fea_onehot, dtype=torch.float32)
                pad_arr = torch.zeros((max_len - len(fea_onehot), 4))
                fea_onehot_pad = torch.cat((fea_onehot, pad_arr), 0)
                dim1.append(fea_onehot_pad)
            dim11 = torch.stack(dim1)
            dim2.append(dim11)
            dim1 = []

        feature = torch.stack(dim2)

        label_list = []
        for i in range(len(label_batch)):
            label_arr = np.array(list(label_batch[i])).reshape(-1, 1)
            label_l_enc = enc.fit_transform(label_arr).toarray()
            label_l_enc = torch.tensor(label_l_enc)
            label_list.append(label_l_enc)

        label = torch.stack(label_list).argmax(dim=2)

        return feature, label

