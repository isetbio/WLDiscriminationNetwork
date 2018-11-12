import nibabel as nib
import numpy as np
import os
from skimage.measure import block_reduce
from dipy.reconst.dti import fractional_anisotropy
from scipy.ndimage import zoom
import zipfile
import shutil

def six2mat(voxel6):
    return voxel6[[0, 1, 3, 1, 2, 4, 3, 4, 5]].reshape(3, 3)


def mat2six(voxelMat):
    return voxelMat.flatten()[[0, 1, 4, 2, 5, 8]]


def getTensorList_general(tensorDir, giveNames=False, crop=[slice(None, None), slice(None, None), slice(None, None)], resizeFactor=1):
    tensors = [os.path.join(tensorDir, t) for t in os.listdir(tensorDir)]
    tensorList = []
    for t in tensors:
        arr = nib.load(t).get_fdata().squeeze()
        arr = arr[crop[0], crop[1], crop[2]]
        arr = block_reduce(arr, block_size=(resizeFactor, resizeFactor, resizeFactor, 1), func=np.mean)
        tensorList.append(arr)
    print(f"Added {len(tensorList)} resized tensors to tensorlist")
    if giveNames:
        return tensorList, tensors
    else:
        return tensorList

def getArrayFromNifti(pathNifti):
    arr = nib.load(pathNifti).get_fdata().squeeze()
    name = os.path.basename(pathNifti).split(".")[0]
    return arr, name


def getFaArrRaw(t):
    arr = t
    matArr = np.apply_along_axis(six2mat, -1, arr)
    evalArr, evecArr = np.linalg.eig(matArr)
    faArr = fractional_anisotropy(evalArr)
    return [faArr, evalArr, evecArr]


def getEncodeArray(arr):
    faArrRaw = getFaArrRaw(arr)
    fa, eval, evec = faArrRaw[0], faArrRaw[1], faArrRaw[2]
    result = np.copy(eval)
    for j in range(fa.shape[0]):
        for k in range(fa.shape[1]):
            for l in range(fa.shape[2]):
                vector = evec[j,k,l][eval[j,k,l].argmax()]
                result[j,k,l] = vector*fa[j,k,l]
    return result


def normalizeByIndividualMean(arr):
    result = []
    for t in arr:
        t = t/np.mean(t)
        result.append(t)
    return result


def resizeTensors(tensors, wantedShape):
    resizedTensors = []
    count = 0
    oldShapes = []
    newShapes = []
    print("Resizing..")
    for t in tensors:
        if t.shape != wantedShape:
            count += 1
            oldShapes.append(t.shape)
            zoomFactor = []
            for i, s in enumerate(t.shape):
                zoomFactor.append(wantedShape[i]/s)
            t = zoom(t, zoomFactor)
            newShapes.append(t.shape)
        resizedTensors.append(t)
    for i in range(count):
        print(f"{oldShapes[i]} -> {newShapes[i]}")
    print(f"Resized {count} out of {len(tensors)} tensors. (This takes quite long)")
    return resizedTensors


def cropBlockResize(tensors, resizeFactor, crop):
    result = []
    rf = resizeFactor
    for t in tensors:
        t = t[crop]
        t = block_reduce(t, block_size=(rf, rf, rf, 1), func=np.mean)
        result.append(t)

    return result

def extractNiftiFromZipArchive(pathZipArchive):
    z = zipfile.ZipFile(pathZipArchive, "r")
    tensorName = [x for x in z.namelist() if x.endswith('tensors.nii.gz') ][0]
    tensorDest = os.path.join(os.path.dirname(pathZipArchive), f"{os.path.basename(pathZipArchive).split('.')[0]}_tensors.nii.gz")
    with z.open(tensorName) as zf, open(tensorDest, 'wb') as f:
        shutil.copyfileobj(zf, f)
    return tensorDest
