import scipy.io as sio
import numpy as np
import torch

def getMatData(pathMat, shuffle=False, extraNoise=False):
    matData = sio.loadmat(pathMat)
    if extraNoise:
        dataNoise = np.transpose(matData['imgNoiseNoStimulus'], (2, 0, 1))
        dataSignal = np.transpose(matData['imgNoiseStimulus'], (2, 0, 1))
    else:
        dataNoise = np.transpose(matData['imgNoNoiseNoStimulus'], (2, 0, 1))
        dataSignal = np.transpose(matData['imgNoNoiseStimulus'], (2, 0, 1))
    labelNoise = np.zeros(dataNoise.shape[0])
    labelSignal = np.ones(dataSignal.shape[0])
    data = np.concatenate([dataNoise, dataSignal])
    labels = np.concatenate([labelNoise, labelSignal])
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels


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
