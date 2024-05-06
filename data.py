import numpy as np
import torch
import data_loader
import data_loader_2a
import data_loader_GIST
from torch.utils.data import Dataset
import pdb
import random

class EEGsignal(Dataset):
    # BCI4_2b dataset
    def __init__(self, train, test, path, model, few, n_iter, is_training, re=None, key_num=None, dataset=None):
        self.train = train
        self.test = test
        self.path = path
        self.model = model
        self.is_training = is_training
        self.re = re
        self.key_num = key_num
        self.few = few
        self.n_iter = n_iter
        if len(self.train) == 9:
            self.train.insert(0, '0')
        if test == [0]:
            if dataset == 'BCI4_2a':
                self.test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            if dataset == 'GIST':
                self.test = list(range(1, 55))
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}
        self.test_key = []
        for subjectnum in list(range(1, 55)):
            if self.is_training == 'train' or self.is_training == 'val':
                if subjectnum in self.train:
                    if dataset == 'BCI4_2b':
                        data, label = data_loader.data_load(self.path, subjectnum, 'train', self.model)
                    if dataset == 'BCI4_2a':
                        data, label = data_loader_2a.data_load(self.path, subjectnum, 'train', self.model)
                    if dataset == 'GIST':
                        data, label = data_loader_GIST.data_load(self.path, subjectnum, 'train', self.model)
                    key = list(data.keys())
                    for i in range(len(key)):
                        c1 = []
                        c2 = []
                        for idx, j in enumerate(label[key[i]]):
                            if j[0] == 1:
                                c1.append(idx)
                            else:
                                c2.append(idx)
                        val_c1 = np.ndarray.tolist(np.random.choice(c1, int(len(c1)*0.2), replace=False))
                        val_c2 = np.ndarray.tolist(np.random.choice(c2, int(len(c2) * 0.2), replace=False))
                        train_c1 = [idx for idx in c1 if not idx in val_c1]
                        train_c2 = [idx for idx in c2 if not idx in val_c2]
                        train_sample = {}
                        train_sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        train_sample['label'] = label[key[i]]
                        train_sample['c1'] = train_c1
                        train_sample['c2'] = train_c2
                        self.train_dict[key[i]] = train_sample
                        val_sample = {}
                        val_sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        val_sample['label'] = label[key[i]]
                        val_sample['c1'] = val_c1
                        val_sample['c2'] = val_c2
                        self.val_dict[key[i]] = val_sample
                    if dataset == 'BCI4_2b':
                        data, label = data_loader.data_load(self.path, subjectnum, 'test', self.model)
                    if dataset == 'BCI4_2a':
                        data, label = data_loader_2a.data_load(self.path, subjectnum, 'test', self.model)
                    if dataset == 'GIST':
                        data, label = data_loader_GIST.data_load(self.path, subjectnum, 'test', self.model)
                    key = list(data.keys())
                    for i in range(len(key)):
                        c1 = []
                        c2 = []
                        for idx, j in enumerate(label[key[i]]):
                            if j[0] == 1:
                                c1.append(idx)
                            else:
                                c2.append(idx)
                        val_c1 = np.ndarray.tolist(np.random.choice(c1, int(len(c1) * 0.2), replace=False))
                        val_c2 = np.ndarray.tolist(np.random.choice(c2, int(len(c2) * 0.2), replace=False))
                        train_c1 = [idx for idx in c1 if not idx in val_c1]
                        train_c2 = [idx for idx in c2 if not idx in val_c2]
                        train_sample = {}
                        train_sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        train_sample['label'] = label[key[i]]
                        train_sample['c1'] = train_c1
                        train_sample['c2'] = train_c2
                        self.train_dict[key[i]] = train_sample
                        val_sample = {}
                        val_sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        val_sample['label'] = label[key[i]]
                        val_sample['c1'] = val_c1
                        val_sample['c2'] = val_c2
                        self.val_dict[key[i]] = val_sample
            if self.is_training == 'test':
                if subjectnum in self.test:
                    if dataset == 'BCI4_2b':
                        data, label = data_loader.data_load(self.path, subjectnum, 'train', self.model)
                    if dataset == 'BCI4_2a':
                        data, label = data_loader_2a.data_load(self.path, subjectnum, 'train', self.model)
                    if dataset == 'GIST':
                        data, label = data_loader_GIST.data_load(self.path, subjectnum, 'train', self.model)
                    key = list(data.keys())
                    self.test_key += key
                    for i in range(len(key)):
                        sample = {}
                        sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        sample['label'] = label[key[i]]
                        c1 = []
                        c2 = []
                        for idx, j in enumerate(label[key[i]]):
                            if j[0] == 1:
                                c1.append(idx)
                            else:
                                c2.append(idx)
                        sample['c1'] = c1
                        sample['c2'] = c2
                        self.test_dict[key[i]] = sample
                    if dataset == 'BCI4_2b':
                        data, label = data_loader.data_load(self.path, subjectnum, 'test', self.model)
                    if dataset == 'BCI4_2a':
                        data, label = data_loader_2a.data_load(self.path, subjectnum, 'test', self.model)
                    if dataset == 'GIST':
                        data, label = data_loader_GIST.data_load(self.path, subjectnum, 'test', self.model)
                    key = list(data.keys())
                    self.test_key += key
                    for i in range(len(key)):
                        sample = {}
                        sample['data'] = np.transpose(data[key[i]][:, :875, :3], [0, 3, 1, 2])
                        sample['label'] = label[key[i]]
                        c1 = []
                        c2 = []
                        for idx, j in enumerate(label[key[i]]):
                            if j[0] == 1:
                                c1.append(idx)
                            else:
                                c2.append(idx)
                        sample['c1'] = c1
                        sample['c2'] = c2
                        self.test_dict[key[i]] = sample
        self.train_key = list(self.train_dict.keys())
        self.train_sample = None

        if is_training == 'test':
            self.sup_list = []
            for i in self.test_key:
                if dataset == 'BCI4_2b':
                    self.sup_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20/'
                                                 +i.split('.npy')[0]+'_sup_list.npy'))
                if dataset == 'BCI4_2a':
                    self.sup_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20_2a/'
                                                 +i.split('.npy')[0]+'_sup_list.npy'))
                if dataset == 'GIST':
                    self.sup_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20_GIST/'
                                                 +i.split('.npy')[0]+'_sup_list.npy'))
            self.que_list = []
            for i in self.test_key:
                if dataset == 'BCI4_2b':
                    self.que_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20/'
                                                 +i.split('.npy')[0]+'_que_list.npy'))
                if dataset == 'BCI4_2a':
                    self.que_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20_2a/'
                                                 +i.split('.npy')[0]+'_que_list.npy'))
                if dataset == 'GIST':
                    self.que_list.append(np.load('/media/NAS/nas_187/sion/checkpoint/EEG_MI_CLASSIFICATION/BCI4_2b/torch/support_list/result_RN_20_GIST/'
                                                 +i.split('.npy')[0]+'_que_list.npy'))

    def sampling(self, data, keys = None):
        if not keys:
            random_key = random.choice(np.sort(list(data.keys())))
        else:
            random_key = keys
        s1_idx = data[random_key]['c1']
        s2_idx = data[random_key]['c2']

        s1 = np.random.choice(s1_idx, self.few, replace=True)
        s2 = np.random.choice(s2_idx, self.few, replace=True)
        if self.is_training == 'train':
            if len(s1_idx+s2_idx) < 16:
                q = s1_idx+s2_idx
            else:
                q = np.random.choice(s1_idx+s2_idx, 16, replace = False)
        else:
            q = random.choice(s1_idx+s2_idx)

        s1 = data[random_key]['data'][s1]
        s2 = data[random_key]['data'][s2]

        if self.is_training == 'train':
            q_label = np.argmax(data[random_key]['label'][q], axis = 1)
        else:
            q_label = np.zeros(1)
            q_label[0] = np.argmax(data[random_key]['label'][q])
        q = data[random_key]['data'][q]

        return s1, s2, q, q_label

    def __len__(self):
        if self.is_training != 'test':
            return self.n_iter
        else:
            return self.que_list[self.key_num].shape[0]

    def __getitem__(self, idx):
        if self.is_training == 'train':
            s1, s2, q, q_label = self.sampling(self.train_dict, self.train_sample)
            s = np.stack((s1, s2), axis=-1)

            return torch.FloatTensor(s), torch.FloatTensor(q), torch.FloatTensor(q_label)
        if self.is_training == 'val':
            s1, s2, q, q_label = self.sampling(self.val_dict, None)
            s = np.stack((s1, s2), axis=-1)

            return torch.FloatTensor(s), torch.FloatTensor(q), torch.FloatTensor(q_label)
        if self.is_training == 'test':
            s1 = self.test_dict[self.test_key[self.key_num]]['data'][self.sup_list[self.key_num][:self.few, self.re, 0]]
            s2 = self.test_dict[self.test_key[self.key_num]]['data'][self.sup_list[self.key_num][:self.few, self.re, 1]]
            q = self.test_dict[self.test_key[self.key_num]]['data'][self.que_list[self.key_num][idx, self.re]]
            q_label = np.zeros(1)
            q_label[0] = np.argmax(
                self.test_dict[self.test_key[self.key_num]]['label'][self.que_list[self.key_num][idx, self.re]])
            s = np.stack((s1, s2), axis=-1)
            key_name = self.test_key[self.key_num]

            return torch.FloatTensor(s), torch.FloatTensor(q), torch.FloatTensor(q_label), key_name

