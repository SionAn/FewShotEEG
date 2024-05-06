import numpy as np
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, axis = -1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = sig.filtfilt(b, a, data, axis=axis)
    y = sig.lfilter(b, a, data, axis = axis)
    return y

def minmax01_norm(x):
    ## dimension is (number, sampling, channel, freq)
    min = np.expand_dims(np.min(x, axis = 1), axis = 1)
    max = np.expand_dims(np.max(x, axis = 1), axis = 1)
    norm = (x-min)/(max-min)

    return norm

def data_load(path, subject, mode, model):
    path = path
    data_list = np.sort(os.listdir(path+'/data/BCI4_2a_raw'))
    label_list = np.sort(os.listdir(path+'/label/BCI4_2a_raw'))
    # subject = subject
    data_dict = {}
    label_dict = {}
    for data_idx in range(len(data_list)):
        if mode == 'train':
            if data_list[data_idx][2] == str(subject) and data_list[data_idx][-5] != 'E':
                print("Load", data_list[data_idx])
                data = np.load(path+'/data/BCI4_2a_raw/'+data_list[data_idx])
                label = np.load(path + '/label/BCI4_2a_raw/' + data_list[data_idx])
                count = 0
                for j in range(data.shape[0]):
                    if np.argmax(label[j]) < 2:
                        count += 1

                data_two = np.zeros((count, 3, data.shape[2]))
                label_two = np.zeros((count, 2))
                count = 0
                for j in range(data.shape[0]):
                    if np.argmax(label[j]) < 2:
                        data_two[count] = data[j, [7, 9, 11]]
                        label_two[count, int(np.argmax(label[j]))] = 1
                        count += 1

                data = data_two
                label = label_two

                data = np.expand_dims(np.swapaxes(data, 1, 2), axis = 3)

                if model == 'HS_CNN':
                    data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                    data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                    data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                    data = np.concatenate((data_47, data_813, data_1332), axis=-1)

                #for j in range(data.shape[0]):
                #    mean = np.mean(data[j])
                #    std = np.std(data[j])
                #    data[j] = (data[j] - mean) / std
                #    max = np.max(data[j])
                #    min = np.min(data[j])
                #    data[j] = (data[j] - min) / (max - min)
                for j in range(data.shape[0]):
                    for k in range(data.shape[2]):
                        for l in range(data.shape[-1]):
                            data[j, :, k, l] = data[j, :, k, l] - np.mean(data[j, :, k, l])
                data_dict[data_list[data_idx]] = data * 1e+7
                label_dict[data_list[data_idx]] = label

        if mode == 'test':
            if data_list[data_idx][2] == str(subject) and data_list[data_idx][-5] == 'E':
                print("Load", data_list[data_idx])
                data = np.load(path + '/data/BCI4_2a_raw/' + data_list[data_idx])
                label = np.load(path + '/label/BCI4_2a_raw/' + data_list[data_idx])

                count = 0
                for j in range(data.shape[0]):
                    if np.argmax(label[j]) < 2:
                        count += 1

                data_two = np.zeros((count, 3, data.shape[2]))
                label_two = np.zeros((count, 2))
                count = 0
                for j in range(data.shape[0]):
                    if np.argmax(label[j]) < 2:
                        data_two[count] = data[j, [7, 9, 11]]
                        label_two[count, int(np.argmax(label[j]))] = 1
                        count += 1

                data = data_two
                label = label_two

                data = np.expand_dims(np.swapaxes(data, 1, 2), axis=3)

                if model == 'HS_CNN':
                    data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                    data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                    data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                    data = np.concatenate((data_47, data_813, data_1332), axis=-1)

                #for j in range(data.shape[0]):
                #    mean = np.mean(data[j])
                #    std = np.std(data[j])
                #    data[j] = (data[j] - mean) / std
                #    max = np.max(data[j])
                #    min = np.min(data[j])
                #    data[j] = (data[j] - min) / (max - min)
                for j in range(data.shape[0]):
                    for k in range(data.shape[2]):
                        for l in range(data.shape[-1]):
                            data[j, :, k, l] = data[j, :, k, l] - np.mean(data[j, :, k, l])
                data_dict[data_list[data_idx]] = data * 1e+7
                label_dict[data_list[data_idx]] = label

    return data_dict, label_dict
