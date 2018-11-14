import scipy.io as sio
import numpy as np
import h5py
import torch

def getMatData(pathMat, shuffle=False):
    matData = sio.loadmat(pathMat)
    data = np.transpose(matData['imgNoise'], (2, 0, 1))
    labels = matData['imgNoiseLabels'].squeeze()
    meanData = np.transpose(matData['meanImg'], (2, 0, 1))
    meanDataLabels = matData['meanImgLabels'].squeeze()
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels, meanData, meanDataLabels


def getH5Data(pathMat, shuffle=False):
    h5Data = h5py.File(pathMat)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    data = h5Dict['imgNoise']
    labels = h5Dict['imgNoiseLabels']
    meanData = h5Dict['meanImg']
    meanDataLabels = h5Dict['meanImgLabels']
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels, meanData, meanDataLabels


def matDataLoader(data, labels, batchSize, shuffle=True):
    epochData = data.clone()
    epochLabels = labels.clone()
    if shuffle:
        selector = np.random.permutation(len(epochData))
        epochData = epochData[selector]
        epochLabels = epochLabels[selector]
    sz = epochData.shape[0]
    i, j = 0, batchSize
    while j < sz + batchSize:
        batchData = epochData[i:min(j, sz)]
        batchLabels = epochLabels[i:min(j, sz)]
        yield batchData, batchLabels
        i += batchSize
        j += batchSize
