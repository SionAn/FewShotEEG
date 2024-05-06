import numpy as np
import os
import scipy.signal as sig
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import pdb
import resampy

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, axis = -1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.filtfilt(b, a, data, axis=axis)
    return y

def minmax01_norm(x):
    ## dimension is (number, sampling, channel, freq)
    min = np.expand_dims(np.min(x, axis = 1), axis = 1)
    max = np.expand_dims(np.max(x, axis = 1), axis = 1)
    norm = (x-min)/(max-min)

    return norm


def data_load(path, subject, model):
    path = path
    data_list = os.listdir(path+'/data/GIST')
    label_list = os.listdir(path+'/label/GIST')
    data_list.sort()
    label_list.sort()
    subject = [subject]
    data_dict = {}
    label_dict = {}
    channel = [12, 47, 49] # C3, Cz, C4
    for i in data_list:
        if int(os.path.splitext(i)[0].split('s')[1]) in subject:
            data = np.load(os.path.join(path, 'data', 'GIST', i)) # [N, C, T]
            if 'HS_CNN' in model:
                data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                data = np.concatenate((data_47, data_813, data_1332), axis=-1)
            for j in range(data.shape[0]):
                for k in range(data.shape[2]):
                    for l in range(data.shape[-1]):
                        data[j, :, k, l] = data[j, :, k, l] - np.mean(data[j, :, k, l])

            data_dict[i] = data[:, 500:500+875] * 1e-2
            label_dict[i] = np.load(path + '/label/GIST/' + i)

    return data_dict, label_dict

if __name__ == '__main__':
    path = '/media/NAS/nas_187/sion/Dataset/EEG/numpy'
    a, b = data_load(path, [1, 2, 3, 4, 5], 'EEGNet')