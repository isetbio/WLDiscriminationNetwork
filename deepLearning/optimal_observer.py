from scipy.stats import poisson
import scipy.io as sio
import numpy as np


def getPoissonAccuracy(dataSignal, dataNoSignal, pureSignal, pureNoSignal):
    allAccuracies = []
    for signal in dataSignal:
        llSignal = poisson.logpmf(signal, pureSignal).sum()
        llNoSignal = poisson.logpmf(signal, pureNoSignal).sum()
        if llSignal > llNoSignal:
            allAccuracies.append(1)
        else:
            allAccuracies.append(0)
    for noSignal in dataNoSignal:
        llSignal = poisson.logpmf(noSignal, pureSignal).sum()
        llNoSignal = poisson.logpmf(noSignal, pureNoSignal).sum()
        if llSignal < llNoSignal:
            allAccuracies.append(1)
        else:
            allAccuracies.append(0)
    return np.mean(allAccuracies)



pathMat = "data/mat_files/30SamplesPerClass_freq_8_contrast_0_015_11-12-18_19_45.mat"
matData = sio.loadmat(pathMat)
dataSignal = np.transpose(matData['imgNoiseStimulus'], (2, 0, 1))
dataNoSignal = np.transpose(matData['imgNoiseNoStimulus'], (2, 0, 1))
pureSignal = np.transpose(matData['imgNoNoiseStimulus'], (2, 0, 1))[0]
pureNoSignal = np.transpose(matData['imgNoNoiseNoStimulus'], (2, 0, 1))[0]

accuray = getPoissonAccuracy(dataSignal, dataNoSignal, pureSignal, pureNoSignal)

print(f"The optimal observer has {accuray*100:.2f}% accuracy on the data\n"
      f"({pathMat})")
