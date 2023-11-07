# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:58:40 2021

@author: hudew
"""


import os, sys
#表示最大优先级，sys.path.insert()加入的是临时搜索路径，程序退出后失效
sys.path.insert(0,'E:\\tools\\')
#import util
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pydicom as dicom
import scipy.io as scio


#%%

class GetCT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

        self.augment = None

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT=dicom.read_file(self.data_names[index])
        data_array = dataCT.pixel_array.astype(np.float32) * dataCT.RescaleSlope + dataCT.RescaleIntercept
        data_array = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array)) ### 0-1
        data_array = np.expand_dims(data_array,2)
        
        data_array_10 = np.tile(data_array,(1,1,1))
        data_array_10= data_array_10.transpose((2,0,1))
        
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)
class GetCTMAT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

        self.augment = None

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT=scio.loadmat(self.data_names[index])['input_5e3']
        #label=scio.loadmat(self.data_names[index])['gt']
        
        data_array = dataCT.astype(np.float32)
        #data_array = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array)) ### 0-1
        data_array = np.expand_dims(data_array,2)
        
        data_array_10 = np.tile(data_array,(1,1,1))
        data_array_10= data_array_10.transpose((2,0,1))
        
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)
        
class DRMAT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

        self.augment = None

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT=scio.loadmat(self.data_names[index])['block']
        #label=scio.loadmat(self.data_names[index])['gt']
        
        data_array = dataCT.astype(np.float32)
        data_array = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array)) ### 0-1
        data_array = np.expand_dims(data_array,2)
        
        data_array_10 = np.tile(data_array,(1,1,1))
        data_array_10= data_array_10.transpose((2,0,1))
        data_array_10 = torch.FloatTensor(data_array_10).unsqueeze_(0)
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)
class BRAINMAT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

        self.augment = None
        #注意是在复数网络GPU问题所以随机裁剪的

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT=scio.loadmat(self.data_names[index])['inputs']
        #label=scio.loadmat(self.data_names[index])['gt']
        
        data_array = dataCT.astype(np.float32)
        data_array = np.expand_dims(data_array, 2)

        data_array = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array)) ### 0-1
        #data_array = np.expand_dims(data_array,2)

        data_array_10 = np.tile(data_array,(1,1,1))
        data_array_10 = data_array_10.transpose((2,0,1))
        data_array_10 = torch.FloatTensor(data_array_10).unsqueeze_(0)
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)
# class trainDataset(Dataset):
    
#     data_dir = "E:\\HumanData\\"
    
#     def __init__(self):
#         super().__init__()
#         self.vol_dir = []
#         self.data = []
        
#         for folder in os.listdir(trainDataset.data_dir):
#             if folder.startswith('ONH'):
#                 file_name = 'SF_'+folder+'.nii.gz'
#                 self.vol_dir.append(trainDataset.data_dir+folder+"\\"+file_name)
        
#         for i in range(len(self.vol_dir)):
#             vol = util.nii_loader(self.vol_dir[i])
#             nslc,h,w = vol.shape
            
#             for j in range(nslc):
#                 im = np.zeros([512,512],dtype=np.float32)
#                 im = util.ImageRescale(vol[j,:,:],[-1,1])
#                 self.data.append(im)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         x = torch.tensor(x).type(torch.FloatTensor)
#         x = x[None,:,:]
#         return x
    
def load_train_data(*, batch_size):
    #dataset = trainDataset()
    
    #dataset = DRMAT(root= "/opt/data/private/zhangzhx2/Anoddpm_complex/jizhu_log1024",augment=None)
    dataset = BRAINMAT(root="/opt/data/private/zhangzhx2/Anoddpm_complex/datasets-brain", augment=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    return loader



# class testDataset(Dataset):
    
#     data_dir = "E:\\HumanData\\"
    
#     def __init__(self, region, snr, idx):
#         super().__init__()
#         self.region = region
#         self.snr = snr
#         self.idx = idx
#         self.vol_dir = []
#         self.data = []
        
#         assert (self.region == 'Fovea' or self.region == 'ONH')
#         assert (self.snr == '92' or self.snr == '96' or self.snr == '101')
        
#         for folder in os.listdir(testDataset.data_dir):
#             vol_reg, _, vol_snr, vol_idx = folder.split('_')
#             if self.region == vol_reg and self.snr == vol_snr and self.idx == vol_idx:
#                 file = "HN_"+folder+".nii.gz"
#                 self.vol_dir.append(testDataset.data_dir+folder+"\\"+file)
        
#         for i in range(len(self.vol_dir)):   
#             vol = util.nii_loader(self.vol_dir[i])
#             _,nslc,h,w = vol.shape
            
#             for j in range(nslc):
#                 im = np.zeros([512,512],dtype=np.float32)
#                 im[:,:500] = util.ImageRescale(vol[0,j,:,:],[-1,1])
#                 self.data.append(im)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         x = torch.tensor(x).type(torch.FloatTensor)
#         x = x[None,:,:]
#         return x

# def load_test_data(*, region, snr, idx, batch_size, shuffle=False):
#     dataset = testDataset(region, snr, idx)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return loader

def load_test_data(*, batch_size):
    #dataset = trainDataset()
    #dataset= BRAINMAT(root= "/data/wangyy/DR_data/linshi_test",augment=None)
    dataset= DRMAT(root= "/opt/data/private/zhangzhx2/Anoddpm_complex/jizhu_log1024_test/",augment=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

#%%
# data_dir = "E:\\HumanData\\ONH_SNR_101_1\\"
# #
# vol = util.nii_loader(data_dir+"SF_ONH_SNR_101_1.nii.gz")     
