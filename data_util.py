'''
Date: 2021-11-28 12:19:06
LastEditors: Liu Yahui
LastEditTime: 2022-07-03 06:27:19
'''
# Reference0: https://github.com/antao97/dgcnn.pytorch/blob/master/data.py
# Reference1: https://github.com/tiangexiang/CurveNet/blob/main/core/data.py
# Reference2: https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/ScanObjectNN.py


import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# change this to your data root
DATA_DIR = './data/'

def download_modelnet40():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        os.mkdir(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'))
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_scanobjectnn():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        os.mkdir(os.path.join(DATA_DIR, 'h5_files'))
        www = 'https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_modelnet40(data_dir, partition):
    # download_modelnet40()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_scanobjectnn(data_dir, partition):
    # download_scanobjectnn()
    h5_name = os.path.join(data_dir, 'h5_files/main_split/', '%s_objectdataset_augmentedrot_scale75.h5'%partition)

    f = h5py.File(h5_name, 'r')
    all_data = f['data'][:].astype('float32')
    all_label = f['label'][:].astype('int64')
    f.close()

    return all_data, all_label


def normalize_pointcloud(pointcloud):
    '''
    Normalize point cloud to a unit sphere at origin
    '''
    pointcloud -= pointcloud.mean(axis=0)
    pointcloud /= np.max(np.linalg.norm(pointcloud, axis=1))
    return pointcloud


def translate_pointcloud(pointcloud):
    '''
    Randomly scale and shift point cloud
    '''
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    '''
    Randomly jitter point cloud
    '''
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    '''
    Randomly rotate point cloud along z-axis
    
    ps: if rotate along x-axis:
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype='float32')

    '''
    angle_z = np.random.uniform(-1, 1) * np.pi
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype='float32')
    pointcloud = np.dot(pointcloud, R_z) 
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    # pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_dir=DATA_DIR, num_points=1024, partition='train'):
        self.data, self.label = load_modelnet40(data_dir, partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN(Dataset):
    def __init__(self, data_dir=DATA_DIR, num_points=1024, partition='training'):
        self.data, self.label = load_scanobjectnn(data_dir, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        

# if __name__ == '__main__':
    # test = ModelNet40(partition='test')
    # data, label = test[0]
    # print(data.shape)  # (1024, 3)
    # print(label.shape) # (1,)

    # test = ShapeNetPart(partition='test')
    # data, label, seg = test[0]
    # print(data.shape)  # (2048, 3)
    # print(label.shape) # (1,)
    # print(seg.shape)   # (2048,)


    # test = S3DIS(partition='test')
    # data, seg = test[0]
    # print(data.shape)  # (4096, 9)
    # print(seg.shape)   # torch.Size([4096])