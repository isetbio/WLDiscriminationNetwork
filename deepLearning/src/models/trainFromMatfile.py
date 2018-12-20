from deepLearning.src.models.resnet_train_test import trainPoisson, test
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import getOptimalObserverAccuracy, calculateDiscriminabilityIndex, getOptimalObserverHitFalsealarm, getOptimalObserverAccuracy_parallel
from deepLearning.src.data.mat_data import getH5MeanData, poissonNoiseLoader
from deepLearning.src.data.logger import Logger
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sys
import os
import csv
import time
import datetime
import pickle


def autoTrain_Resnet_optimalObserver(pathMat, device=None, lock=None, train_nn=False, includeShift=False):
    # relevant variables
    startTime = time.time()
    print(device, pathMat)
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    test_interval = 2
    batchSize = 128
    numSamplesEpoch = 10000
    outPath = os.path.dirname(pathMat)
    fileName = os.path.basename(pathMat).split('.')[0]
    sys.stdout = Logger(f"{os.path.join(outPath, fileName)}_log.txt")
    if includeShift:
        meanData, meanDataLabels, dataContrast, dataShift = getH5MeanData(pathMat, includeContrast=True, includeShift=True)
    else:
        meanData, meanDataLabels, dataContrast = getH5MeanData(pathMat, includeContrast=True)
    # data = torch.from_numpy(data).type(torch.float32)
    # pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
    # data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
    # Image.fromarray(data[4]*(255/20)).show()

    testDataFull, testLabelsFull = poissonNoiseLoader(meanData, size=10000, numpyData=True)
    if len(meanData) > 2:
        accOptimal, optimalOPredictionLabel = getOptimalObserverAccuracy_parallel(testDataFull, testLabelsFull, meanData, returnPredictionLabel=True)
        pickle.dump(optimalOPredictionLabel, open(os.path.join(outPath, "optimalOpredictionLabel.p"), 'wb'))
        pickle.dump(dataContrast, open(os.path.join(outPath, "contrastLabels.p"), 'wb'))
    else:
        accOptimal = getOptimalObserverAccuracy(testDataFull, testLabelsFull, meanData)
    print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")

    d1 = calculateDiscriminabilityIndex(meanData)
    print(f"Theoretical d index is {d1}")

    d2 = getOptimalObserverHitFalsealarm(testDataFull, testLabelsFull, meanData)
    print(f"Optimal observer d index is {d2}")

    testData = testDataFull[:500]
    testLabels = testLabelsFull[:500]
    dimIn = testData[0].shape[1]
    dimOut = len(meanData)

    if train_nn:
        Net = GrayResnet18(dimOut)
        Net.cuda()
        # print(Net)
        # Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
        criterion = nn.NLLLoss()
        bestTestAcc = 0

        # Test the network
        # testAcc = test(batchSize, testData, testLabels, Net, dimIn)
        # Train the network
        epochs = 5
        learning_rate = 0.001
        optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
        testLabels = torch.from_numpy(testLabels.astype(np.long))
        testData = torch.from_numpy(testData).type(torch.float32)
        Net, testAcc = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
        # bestTestAcc = max(bestTestAcc, bestTestAccStep)

        print(f"Best accuracy to date is {bestTestAcc*100:.2f} percent")

        # Train the network more
        epochs = 5
        learning_rate = 0.0001
        optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

        Net, testAcc = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
        # bestTestAcc = max(bestTestAcc, bestTestAccStep)

        # Train the network more
        epochs = 5
        learning_rate = 0.00001
        optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

        Net, testAcc = trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
        # bestTestAcc = max(bestTestAcc, bestTestAccStep)
        torch.save(Net.state_dict(), os.path.join(outPath, f"resNet_weights_{fileName}.torch"))
        print("saved resNet weights to", f"resNet_weights_{fileName}.torch")
        testLabelsFull = torch.from_numpy(testLabelsFull.astype(np.long))
        testDataFull = torch.from_numpy(testDataFull).type(torch.float32)
        testAcc, nnPredictionLabels = test(batchSize, testDataFull, testLabelsFull, Net, dimIn, includePredictionLabels=True)
        pickle.dump(nnPredictionLabels, open(os.path.join(outPath, "nnPredictionLabels.p"), 'wb'))
    else:
        testAcc = 0.5


    print(f"ResNet accuracy is {testAcc*100:.2f}%")
    print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
    print(f"Optimal observer d index is {d2}")
    print(f"Theoretical d index is {d1}")

    if lock is not None:
        lock.acquire()
    resultCSV = os.path.join(outPath, "results.csv")
    file_exists = os.path.isfile(resultCSV)

    with open(resultCSV, 'a') as csvfile:
        if not includeShift:
            headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast']
            writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': dataContrast[0].astype(np.float64)})
        else:
            headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast', 'shift']
            writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': dataContrast[0].astype(np.float32), 'shift': dataShift[1].astype(np.float64)})

    print(f'Wrote results to {resultCSV}')
    if lock is not None:
        lock.release()
    endTime = time.time()

    print(f"done! It took {str(datetime.timedelta(seconds=endTime-startTime))} hours:min:seconds")
    sys.stdout = sys.stdout.revert()
