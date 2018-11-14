from deepLearning.src.models.resnet_train_test import train
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import getOptimalObserverAccuracy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import numpy as np
from PIL import Image
import pickle
from scipy.stats import lognorm
import torchvision.models as models

from deepLearning.src.data.mat_data import getMatData, matDataLoader

# relevant variables
test_interval = 2
batchSize = 128
pathMat = "data/mat_files/25_samplesPerClass_freq_1-2-3-4-5-6-7-8-9_contrast_0_01_11-13-18_15_28.mat"

data, labels, meanData, meanDataLabels = getMatData(pathMat, shuffle=True)
# data = torch.from_numpy(data).type(torch.float32)
# pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
# data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
# Image.fromarray(data[4]*(255/20)).show()

dimIn = data[0].shape[1]
dimOut = len(meanData)
labels = torch.from_numpy(labels.astype(np.long))
testData = data[:int(len(data)*0.2)]
testLabels = labels[:int(len(data)*0.2)]
trainData = data[int(len(data)*0.2):]
trainLabels = labels[int(len(data)*0.2):]

accOptimal = getOptimalObserverAccuracy(testData, testLabels, meanData)
print(accOptimal)

Net = GrayResnet18(dimOut, dropout=0.8)
Net.cuda()
print(Net)
# Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
criterion = nn.NLLLoss()


# Train the network
epochs = 50
learning_rate = 0.001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

Net, bestTestAcc = train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
print(f"Best accuracy to date is {bestTestAcc*100:.2f} percent")
# Train the network more
epochs = 50
learning_rate = 0.0001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate, amsgrad=True)

Net, bestTestAcc = train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
print(f"Best accuracy is {bestTestAcc*100:.2f} percent")


# torch.save(Net.state_dict(), "trained_ResNet_with_noise.torch")

print('\nDone!')
