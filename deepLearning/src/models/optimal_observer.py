from scipy.stats import poisson, norm
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


def getOptimalObserverAccuracy(testData, testLabels, meanData):
    allAccuracies = []
    for datum, label in zip(testData, testLabels):
        llVals = []
        for meanDatum in meanData:
            llVals.append(poisson.logpmf(datum, meanDatum).sum())
        prediction = np.argmax(llVals)
        allAccuracies.append(prediction == label)
        # print(f"prediction: {prediction}, label is {label}.")
    return np.mean(allAccuracies)

def getOptimalObserverHitFalsealarm(testData, testLabels, meanData):
    hits = []
    falseAlarms = []
    allAccuracies = []
    for datum, label in zip(testData, testLabels):
        llVals = []
        for meanDatum in meanData:
            llVals.append(poisson.logpmf(datum, meanDatum).sum())
        prediction = np.argmax(llVals)
        if label == 1: # signal
            hits.append(label == prediction)
        else:
            falseAlarms.append(label != prediction)
        if label != prediction:
            # print("test")
            pass
        allAccuracies.append(prediction == label)
    d = norm.ppf(np.mean(hits))-norm.ppf(np.mean(falseAlarms))
    return d

def calculateDiscriminabilityIndex(meanData):
    alpha = meanData[0]
    beta = meanData[1]
    d = np.sum((beta-alpha) * np.log(beta/alpha)) / np.sqrt(0.5*np.sum(((alpha+beta) * np.log(beta/alpha)**2)))
    # 8.474092465767908
    return d



if __name__ == '__main__':
    pathMat = "data/mat_files/30SamplesPerClass_freq_8_contrast_0_015_11-12-18_19_45.mat"
    matData = sio.loadmat(pathMat)
    dataSignal = np.transpose(matData['imgNoiseStimulus'], (2, 0, 1))
    dataNoSignal = np.transpose(matData['imgNoiseNoStimulus'], (2, 0, 1))
    pureSignal = np.transpose(matData['imgNoNoiseStimulus'], (2, 0, 1))[0]
    pureNoSignal = np.transpose(matData['imgNoNoiseNoStimulus'], (2, 0, 1))[0]

    accuray = getPoissonAccuracy(dataSignal, dataNoSignal, pureSignal, pureNoSignal)

    print(f"The optimal observer has {accuray*100:.2f}% accuracy on the data\n"
          f"({pathMat})")
