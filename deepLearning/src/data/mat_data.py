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
    labels = h5Dict['imgNoiseFreqs']
    meanData = h5Dict['noNoiseImg']
    meanDataLabels = h5Dict['noNoiseImgFreq']
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels, meanData, meanDataLabels


def getH5MeanData(pathMat, includeContrast=False):
    h5Data = h5py.File(pathMat)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    noNoiseData = h5Dict['noNoiseImg']
    noNoiseLabels = h5Dict['noNoiseImgFreq']
    dataContrast = h5Dict['noNoiseImgContrast']
    if includeContrast:
        return noNoiseData, noNoiseLabels, dataContrast
    else:
        return noNoiseData, noNoiseLabels


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


def poissonNoiseLoader(meanData, size, numpyData=False):
    if numpyData:
        labels = np.random.randint(len(meanData), size=size)
        data = []
        for l in labels:
            data.append(np.random.poisson(meanData[l]))
        data = np.stack(data)
        return data, labels
    else:
        labels = torch.randint(len(meanData), (size,)).type(torch.long)
        data = []
        for l in labels:
            data.append(torch.poisson(meanData[l]))
        data = torch.stack(data)
        return data, labels




if __name__ == '__main__':
    pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5"
    meanData, meanDataLabels = getH5MeanData(pathMat)
    data, labels = poissonNoiseLoader(meanData, 16)
    print("done")