from torch.autograd import Variable
import numpy as np
import torch


from deepLearning.src.data.mat_data import poissonNoiseLoader, matDataLoader


def test(batchSize, testData, testLabels, Net, dimIn):
    allAccuracy =[]
    allWrongs = []
    for batch_idx, (data, target) in enumerate(matDataLoader(testData, testLabels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        # data = data.view(-1, dimIn)
        data = data.view(-1, 1, dimIn, dimIn)
        # Net.eval()
        net_out = Net(data)
        prediction = net_out.max(1)[1]
        selector = (prediction != target).cpu().numpy().astype(np.bool)
        wrongs = data_temp[selector]
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        allWrongs.extend(wrongs)
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    return np.mean(allAccuracy)


def train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn):
    bestTestAcc = 0
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        logCount = 0
        testAcc = 0
        for batch_idx, (data, target) in enumerate(matDataLoader(trainData, trainLabels, batchSize, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            data = data.view(-1, 1, dimIn, dimIn)
            optimizer.zero_grad()
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            testAcc = test(batchSize, testData, testLabels, Net, dimIn)
            if testAcc > bestTestAcc:
                bestTestAcc = testAcc
    return Net, testAcc

def trainPoisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn):
    bestTestAcc = 0
    meanData = torch.from_numpy(meanData).type(torch.float32).cuda()
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        logCount = 0
        testAcc = 0
        print(f"One epoch simulates {numSamplesEpoch} samples.")
        for batch_idx in range(int(np.round(numSamplesEpoch/batchSize))):
            data, target = poissonNoiseLoader(meanData, batchSize, numpyData=False)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            data = data.view(-1, 1, dimIn, dimIn)
            optimizer.zero_grad()
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            testAcc = test(batchSize, testData, testLabels, Net, dimIn)
            if testAcc > bestTestAcc:
                bestTestAcc = testAcc
    return Net, testAcc
