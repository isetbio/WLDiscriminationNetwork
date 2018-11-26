from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from glob import glob
import GPUtil
import multiprocessing as mp
import time

deviceIDs = GPUtil.getAvailable(order = 'first', limit = 10, maxLoad = 0.1, maxMemory = 0.1, excludeID=[], excludeUUID=[])
pathMatDir = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/experiment_freq_1_log_contrasts/"

def matFileGen(pathMatDir):
    matFiles = glob(f'{pathMatDir}*.h5')
    matFiles.sort()
    for matFile in matFiles:
        yield matFile


pathGen = matFileGen(pathMatDir)
Procs = {}
while True:
    try:
        if Procs == {}:
            for device in deviceIDs:
                pathMat = next(pathGen)
                print(f"Running {pathMat} on GPU {device}")
                currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=(pathMat, int(device),))
                Procs[str(device)] = currP
                currP.start()
        for device, proc in Procs.items():
            if not proc.is_alive():
                pathMat = next(pathGen)
                print(f"Running {pathMat} on GPU {device}")
                currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=(pathMat, int(device),))
                Procs[str(device)] = currP
                currP.start()
    except StopIteration:
        break

    time.sleep(5)


for proc in Procs.values():
    proc.join()

print("done!")
