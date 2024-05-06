import numpy as np
import os
import scipy.signal as sig
from scipy.linalg import fractional_matrix_power
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
    y = sig.filtfilt(b, a, data, axis=axis)
    return y

def minmax01_norm(x):
    ## dimension is (number, sampling, channel, freq)
    min = np.expand_dims(np.min(x, axis = 1), axis = 1)
    max = np.expand_dims(np.max(x, axis = 1), axis = 1)
    norm = (x-min)/(max-min)

    return norm

def Riemannian_mean(data, index, prev, count):
    C = np.cov(data[index])
    if count == 0:
        M = C
    else:
        M_prev = prev
        M_inv = np.linalg.inv(M_prev)
        cal = fractional_matrix_power(M_inv, 0.5)
        cal = np.matmul(cal, C)
        cal = np.matmul(cal, fractional_matrix_power(M_inv, 0.5))
        cal = fractional_matrix_power(cal, 1/(index+1+count))
        cal = np.matmul(fractional_matrix_power(M_prev, 0.5), cal)
        cal = np.matmul(cal, fractional_matrix_power(M_prev, 0.5))
        M = cal

    return M

def data_load(path, subject, mode, model):
    path = path
    data_list = os.listdir(path+'/data/BCI4_2b')
    label_list = os.listdir(path+'/label/BCI4_2b')
    data_list.sort()
    label_list.sort()
    subject_num = subject
    data_dict = {}
    label_dict = {}
    for i in range(len(data_list)):
        if mode == 'train':
            if data_list[i][2] == str(subject_num) and data_list[i][5] == 'T':
                data = np.load(path+'/data/BCI4_2b/'+data_list[i])
                data = np.expand_dims(np.swapaxes(data, 1, 2), axis = 3)
                data = data[::, ::, 0:3]
                # print(data_list[i], np.min(data*1e+8), np.max(data*1e+8))
                # data_norm = data.copy()
                # for j in range(data.shape[0]):
                #     for k in range(data.shape[2]):
                #         max = np.max(data[j, :, k])
                #         min = np.min(data[j, :, k])
                #         data_norm[j, :, k] = (data[j, :, k] - min) / (max - min)
                # pdb.set_trace()
                if model == 'HS_CNN' or model == 'HS_CNN_IROS':
                    data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                    data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                    data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                    data = np.concatenate((data_47, data_813, data_1332), axis=-1)
                # for j in range(data.shape[0]):
                #     mean = np.mean(data[j])
                #     std = np.std(data[j])
                #     data[j] = (data[j] - mean) / std
                    # max = np.max(data[j])
                    # min = np.min(data[j])
                    # data[j] = (data[j] - min) / (max - min)
                for j in range(data.shape[0]):
                    for k in range(data.shape[2]):
                        for l in range(data.shape[-1]):
                            data[j, :, k, l] = data[j, :, k, l] - np.mean(data[j, :, k, l])
                            # min = np.min(data[j, :, k, l])
                            # max = np.max(data[j, :, k, l])
                            # data[j, :, k, l] = (data[j, :, k, l]-min)/(max-min)
                data_dict[data_list[i]] = data * 1e+8
                label_dict[data_list[i]] = np.load(path + '/label/BCI4_2b/' + label_list[i])

        if mode == 'test':
            if data_list[i][2] == str(subject_num) and data_list[i][5] == 'E':
                data = np.load(path + '/data/BCI4_2b/' + data_list[i])
                data = np.expand_dims(np.swapaxes(data, 1, 2), axis=3)
                data = data[::, ::, 0:3]
                # print(data_list[i], np.min(data * 1e+8), np.max(data * 1e+8))
                if model == 'HS_CNN' or model == 'HS_CNN_IROS':
                    data_47 = (butter_bandpass_filter(data, 4, 7, 250, 1, 4))
                    data_813 = (butter_bandpass_filter(data, 8, 13, 250, 1, 4))
                    data_1332 = (butter_bandpass_filter(data, 13, 32, 250, 1, 4))
                    data = np.concatenate((data_47, data_813, data_1332), axis=-1)
                # for j in range(data.shape[0]):
                #     mean = np.mean(data[j])
                #     std = np.std(data[j])
                #     data[j] = (data[j] - mean) / std
                    # max = np.max(data[j])
                    # min = np.min(data[j])
                    # data[j] = (data[j] - min) / (max - min)
                for j in range(data.shape[0]):
                    for k in range(data.shape[2]):
                        for l in range(data.shape[-1]):
                            data[j, :, k, l] = data[j, :, k, l] - np.mean(data[j, :, k, l])
                            # min = np.min(data[j, :, k, l])
                            # max = np.max(data[j, :, k, l])
                            # data[j, :, k, l] = (data[j, :, k, l] - min) / (max - min)
                data_dict[data_list[i]] = data * 1e+8
                label_dict[data_list[i]] = np.load(path + '/label/BCI4_2b/' + label_list[i])

    return data_dict, label_dict


