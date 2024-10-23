import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fairseq
import lmdb
import shutil


def preprocess_lmdb(out_path,data_path,mode,csvname,reset):
    # 如果输出路径已存在，并且不需要重置数据库，则直接返回
    if os.path.exists(out_path + mode):
        if reset == False:
            return None
        else:
            shutil.rmtree(out_path + mode)  # 如果需要重置数据库，删除已存在的数据库文件夹
    print('start preprocessing .')
    # 从 CSV 文件中读取数据和标签
    data = pd.read_csv(csvname)
    names = data.name.values
    labels = data.label.values
    # 打开 LMDB 数据库
    env = lmdb.open(out_path + mode,map_size = 409951162700)
    count = 0
    ones = np.ones((324),dtype = np.float32)
    # 使用 with 语句创建一个事务
    with env.begin(write = True) as txn:
        for i in range(len(labels)):
            name1 = names[i]
            # 从文件路径读取 WavLM 特征数据
            # data1 = np.load(data_path + 'Session' + name1[4]+'/'+name1+'.npy')
            data1 = np.load(data_path +'/'+name1+'.npy')
            newdata1 = np.zeros((324,1024),dtype = np.float32)
            mask = np.zeros((324),dtype = np.float32)
            lens = data1.shape[0]
            # 处理长度超过 324 的情况
            if lens > 324:
                newlens = 324
            else:
                newlens = lens
                mask[newlens:] = ones[newlens:]
            newdata1[:newlens, :] = data1[:newlens, :]  # 将数据填充到新的数组中
            key_data = 'data-%05d'%count
            key_label = 'label-%05d'%count
            key_mask = 'mask-%05d'%count
            # 将数据、标签和掩码存储到 LMDB 数据库中
            txn.put(key_data.encode(),newdata1)
            txn.put(key_label.encode(),labels[i])
            txn.put(key_mask.encode(),mask)
            count += 1
    env.close()
    print(' preprocess is finished !')

# 设置输出路径
out_path = r'/home/lijiahao/DWFormer-main/new_database_wavlm_mask_324(1)' #save the Dataset
os.mkdir(out_path) # 创建保存数据库的文件夹
# 遍历训练集和验证集
for i in range(1,6):
    csvname1 = r'/home/lijiahao/DWFormer-main/Feature_Extractor/iemocap_data/train'+ str(i) + '.csv' # 保存 WavLM 特征的文件夹
    data_path = r'/home/lijiahao/DWFormer-main/IEMOCAP/Feature/Wavlm/Session'#place where saves the WavLM features
    mode = r'train' + str(i)
    os.mkdir(out_path+mode) # 创建保存数据库的文件夹
    reset = True
    preprocess_lmdb(out_path,data_path,mode,csvname1,reset)
    # 配置验证集的 CSV 文件路径、数据路径和模式
    mode = r'valid' + str(i)
    os.mkdir(out_path+mode) # 创建保存数据库的文件夹
    csvname2 = r'/home/lijiahao/DWFormer-main/Feature_Extractor/iemocap_data/valid' + str(i) + '.csv'
    preprocess_lmdb(out_path,data_path,mode,csvname2,reset)