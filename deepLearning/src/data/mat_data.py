import scipy.io as sio
import numpy as np
import h5py
import torch

def get_mat_data(pathMat, shuffle=False):
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


def get_h5data(pathMat, shuffle=False):
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


def get_h5mean_data(pathMat, includeContrast=False, includeShift=False, includeAngle=False, them_cones=False,
                    separate_rgb=False, meanData_rounding=None, shuffled_pixels=False):
    h5Data = h5py.File(pathMat)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    args = []
    if them_cones:
        # 2 = red, 3 = green, 4 = blue
        mosaic = h5Dict['mosaicPattern']
        img_data = h5Dict['excitationsData']
        if len(img_data.shape) == 4:
            # quickfix different h5 files... (to be deleted)
            if img_data.shape[-1] == 1:
                img_data = np.squeeze(img_data)
            else:
                img_data = np.transpose(img_data, (3, 1, 2, 0))
        else:
            img_data = np.transpose(img_data, (2, 1, 0))
        if separate_rgb:
            if len(img_data.shape) == 4:
                r = np.copy(img_data)
                r[:, mosaic != 2, :] = 0
                g = np.copy(img_data)
                g[:, mosaic != 3, :] = 0
                b = np.copy(img_data)
                b[:, mosaic != 4, :] = 0
                stack_list = []
                for i in range(img_data.shape[3]):
                    stack_list.append(r[..., i])
                    stack_list.append(g[..., i])
                    stack_list.append(b[..., i])
                img_data = np.stack(stack_list, axis=-1)
            else:
                r = np.copy(img_data)
                r[:, mosaic != 2] = 0
                g = np.copy(img_data)
                g[:, mosaic != 3] = 0
                b = np.copy(img_data)
                b[:, mosaic != 4] = 0
                img_data = np.stack((r, g, b), axis=-1)
        # img_data = np.transpose(img_data, (3, 1, 2, 0))
        args.append(img_data)
        args.append(h5Dict['spatialFrequencyCyclesPerDeg'])
        if includeContrast:
            args.append(h5Dict['contrast'])
        if includeShift:
            args.append(h5Dict['shift'])
        if includeAngle:
            args.append(h5Dict['rotation'])
    else:
        # round to .1f experiment:
        experiment = h5Dict['noNoiseImg']
        # rotate 90 degrees
        experiment = np.transpose(experiment, (0, 2, 1))
        # experiment += 0.567891011121314
        if meanData_rounding is not None:
            experiment = np.round(experiment, meanData_rounding)
        # experiment -= 0.567891011121314
        if shuffled_pixels:
            np.random.seed(42)
            rows = experiment.shape[-2]
            cols = experiment.shape[-1]
            shuff_args = np.random.permutation(np.arange(rows*cols))
            result = []
            for md in experiment:
                result.append(md.flatten()[shuff_args].reshape(rows,cols))
            experiment = np.stack(result)
            print("nice")
        args.append(experiment)
        #####################
        # args.append(h5Dict['noNoiseImg'])
        args.append(h5Dict['noNoiseImgFreq'])
        if includeContrast:
            args.append(h5Dict['noNoiseImgContrast'])
        if includeShift:
            args.append(h5Dict['noNoiseImgPhase'])
        if includeAngle:
            args.append(h5Dict['noNoiseImgAngle'])
    return args


def mat_data_loader(data, labels, batchSize, shuffle=True):
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


def poisson_noise_loader(meanData, size, numpyData=False):
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
    meanData, meanDataLabels = get_h5mean_data(pathMat)
    data, labels = poisson_noise_loader(meanData, 16)
    print("done")
