import scipy.io as sio
import numpy as np
import h5py
import torch
from skimage.util import view_as_blocks

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
            print(f"Rounding mean_data to {meanData_rounding} decimals..")
            experiment = np.round(experiment, meanData_rounding)
        # experiment -= 0.567891011121314
        if shuffled_pixels > 0:
            experiment = shuffle_pixels(experiment, shuffled_pixels)
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


def shuffle_pixels(matrices, block_size):
    block_size = int(block_size)
    np.random.seed(42)
    rows = matrices.shape[-2]
    cols = matrices.shape[-1]
    shuffle_rows = rows // block_size
    shuffle_cols = cols // block_size
    padding_rows = rows % block_size
    padding_cols = cols % block_size


    shuff_args = np.random.permutation(np.arange(shuffle_rows * shuffle_cols))
    result = []
    for md in matrices:
        # remove padding
        pad_up = np.ceil(padding_rows/2).astype(np.int)
        pad_down = np.floor(padding_rows/2).astype(np.int)
        pad_left = np.ceil(padding_cols/2).astype(np.int)
        pad_right = np.floor(padding_cols/2).astype(np.int)
        pad_mat_up = md[:pad_up]
        md = md[pad_up:]
        if pad_down == 0:
            pad_mat_down = md[:0]
        else:
            pad_mat_down = md[-pad_down:]
            md = md[:-pad_down]
        pad_mat_left = md[:, :pad_left]
        md = md[:, pad_left:]
        if pad_down == 0:
            pad_mat_right = md[:, :0]
        else:
            pad_mat_right = md[:, -pad_right:]
            md = md[:, :-pad_right]
        md = view_as_blocks(md, (block_size, block_size))
        height = md.shape[0]
        width = md.shape[1]
        res = md.reshape(-1, block_size, block_size)[shuff_args].reshape(height, width, block_size, block_size)
        res = res.transpose(0, 2, 1, 3).reshape(-1, res.shape[1] * res.shape[3])

        # reassemble
        res = np.block([[pad_mat_up], [pad_mat_left, res, pad_mat_right], [pad_mat_down]])
        result.append(res)
    matrices = np.stack(result)
    return matrices


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
    np.random.seed(42)
    if numpyData:
        # more data means all data after #0 are signal cases
        data = []
        if len(meanData) > 2:
            labels = np.random.randint(2, size=size)
            for l in labels:
                if l == 0:
                    data.append(np.random.poisson(meanData[l]))
                else:
                    selector = np.random.randint(1, len(meanData))
                    data.append(np.random.poisson(meanData[selector]))
        else:
            labels = np.random.randint(len(meanData), size=size)
            for l in labels:
                data.append(np.random.poisson(meanData[l]))
        data = np.stack(data)
        return data, labels
    else:
        data = []
        if len(meanData) > 2:
            labels = torch.randint(2, (size,)).type(torch.long)
            for l in labels:
                if l==0:
                    data.append(torch.poisson(meanData[l]))
                else:
                    selector = np.random.randint(1, len(meanData))
                    data.append(torch.poisson(meanData[selector]))
        else:
            labels = torch.randint(len(meanData), (size,)).type(torch.long)
            for l in labels:
                data.append(torch.poisson(meanData[l]))
        data = torch.stack(data)
        return data, labels




if __name__ == '__main__':
    pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5"
    meanData, meanDataLabels = get_h5mean_data(pathMat)
    data, labels = poisson_noise_loader(meanData, 16)
    print("done")
