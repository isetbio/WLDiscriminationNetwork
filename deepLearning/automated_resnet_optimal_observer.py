from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from glob import glob
import time
import datetime

pathMatDir = '/share/wandell/data/reith/matlabData/shift_contrast33/'

matFiles = glob(f'{pathMatDir}*.h5')
matFiles.sort()
programStart = time.time()
for matFile in matFiles:
    if matFile[-5:-3] == 'oo':
        print(f"Only optimal observer for: {matFile}")
        autoTrain_Resnet_optimalObserver(matFile, train_nn=False, includeShift=True)
    else:
        print(matFile)
        autoTrain_Resnet_optimalObserver(matFile, train_nn=True, includeShift=True)


programEnd = time.time()

print(f"Whole program finished! It took {str(datetime.timedelta(seconds=programEnd-programStart))} hours:min:seconds")
print("done!")
