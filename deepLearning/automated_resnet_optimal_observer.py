from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from glob import glob

pathMatDir = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/experiment_shift_contrasts_100/"

matFiles = glob(f'{pathMatDir}*.h5')
matFiles.sort()
for matFile in matFiles:
    if matFile.split('_')[-5] == '0' and int(matFile.split('_')[-4]) <= 10000:
        print(f"Only optimal observer for: {matFile}")
        autoTrain_Resnet_optimalObserver(matFile, train_nn=False)
    else:
        print(matFile)
        autoTrain_Resnet_optimalObserver(matFile, train_nn=True)



print("done!")
