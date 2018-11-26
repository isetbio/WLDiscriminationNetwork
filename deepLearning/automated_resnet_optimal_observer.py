from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from glob import glob

pathMatDir = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/experiment_freq_1_log_contrasts200/"

matFiles = glob(f'{pathMatDir}*.h5')
matFiles.sort()
for matFile in matFiles:
    print(matFile)
    autoTrain_Resnet_optimalObserver(matFile)



print("done!")
