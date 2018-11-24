from deepLearning.src.models.resnet_train_test import trainPoisson
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import getOptimalObserverAccuracy, calculateDiscriminabilityIndex, getOptimalObserverHitFalsealarm
from deepLearning.src.data.mat_data import getH5MeanData, poissonNoiseLoader
from deepLearning.src.data.logger import Logger
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sys
import os
import csv


def autoTrain_Resnet_optimalObserver(pathMat, device=None):
    # relevant variables
    if device is not None:
        torch.cuda.device(device)
    test_interval = 2
    batchSize = 128
    numSamplesEpoch = 10000
    outPath = os.path.dirname(pathMat)
    fileName = os.path.basename(pathMat).split('.')[0]
    sys.stdout = Logger(f"{os.path.join(outPath, fileName)}_log.txt")

    meanData, meanDataLabels, dataContrast = getH5MeanData(pathMat, includeContrast=True)
    # data = torch.from_numpy(data).type(torch.float32)
    # pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
    # data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
    # Image.fromarray(data[4]*(255/20)).show()

    testData, testLabels = poissonNoiseLoader(meanData, size=2000, numpyData=True)
    accOptimal = getOptimalObserverAccuracy(testData, testLabels, meanData)
    print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")

    d1 = calculateDiscriminabilityIndex(meanData)
    print(f"Theoretical d index is {d1}")

    d2 = getOptimalObserverHitFalsealarm(testData, testLabels, meanData)
    print(f"Optimal observer d index is {d2}")

    dimIn = testData[0].shape[1]
    dimOut = len(meanData)


    Net = GrayResnet18(dimOut)
    Net.cuda()
    # print(Net)
    # Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
    criterion = nn.NLLLoss()
    bestTestAcc = 0

    # Test the network
    # testAcc = test(batchSize, testData, testLabels, Net, dimIn)
    # Train the network
    epochs = 4
    learning_rate = 0.001
    optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
    testLabels = torch.from_numpy(testLabels.astype(np.long))
    testData = torch.from_numpy(testData).type(torch.float32)
    Net, bestTestAccStep = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
    bestTestAcc = max(bestTestAcc, bestTestAccStep)

    print(f"Best accuracy to date is {bestTestAcc*100:.2f} percent")

    # Train the network more
    epochs = 4
    learning_rate = 0.0001
    optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

    Net, bestTestAccStep = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
    bestTestAcc = max(bestTestAcc, bestTestAccStep)

    # Train the network more
    epochs = 4
    learning_rate = 0.00001
    optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

    Net, bestTestAccStep = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
    bestTestAcc = max(bestTestAcc, bestTestAccStep)


    print(f"Best ResNet accuracy is {bestTestAcc*100:.2f}%")
    print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
    print(f"Optimal observer d index is {d2}")
    print(f"Theoretical d index is {d1}")

    torch.save(Net.state_dict(), os.path.join(outPath, f"resNet_weights_{fileName}.torch"))
    print("saved resNet weights to", f"resNet_weights_{fileName}.torch")

    resultCSV = os.path.join(outPath, "results.csv")
    file_exists = os.path.isfile(resultCSV)

    with open(resultCSV, 'a') as csvfile:
        headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'ResNet_accuracy': bestTestAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': dataContrast[0].astype(np.float32)})
    print(f'Wrote results to {resultCSV}')
    print("done!")
    sys.stdout = sys.stdout.revert()
