from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from deepLearning.src.models.Resnet import PretrainedResnetFrozen, NotPretrainedResnet
from glob import glob
import GPUtil
import multiprocessing as mp
import time
import datetime



def matfile_gen(pathMatDir):
    matFiles = glob(f'{pathMatDir}**/*.h5', recursive=True)
    matFiles.sort()
    for matFile in matFiles:
        yield matFile


def run_on_folder(dirname, deeper_pls=False, NetClass=None, NetClass_param=None):
    deviceIDs = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    print(deviceIDs)
    function_start = time.time()
    pathGen = matfile_gen(dirname)
    Procs = {}
    lock = mp.Lock()
    while True:
        try:
            if Procs == {}:
                for device in deviceIDs:
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, 'train_nn': True, 'include_shift': True,
                                               'NetClass': NetClass, 'deeper_pls': deeper_pls, 'NetClass_param': NetClass_param})
                    Procs[str(device)] = currP
                    currP.start()
            for device, proc in Procs.items():
                if not proc.is_alive():
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, 'train_nn': True, 'include_shift': True,
                                               'NetClass': NetClass, 'deeper_pls': deeper_pls, 'NetClass_param': NetClass_param})
                    Procs[str(device)] = currP
                    currP.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in Procs.values():
        proc.join()

    function_end = time.time()

    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    time.sleep(60)
    print("done!")


if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/harmonic_shift_calibration_include_shifts/', deeper_pls=False, NetClass=None)
    with open('/share/wandell/data/reith/harmonic_shift_calibration_include_shifts/time.txt', 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")




'''
Older runs for documentation purposes..
##################################################
if __name__ == '__main__':
    full_start = time.time()
    for i in range(1,21):
        run_on_folder(f'/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_{i}/', deeper_pls=False, NetClass=None)
    with open('/share/wandell/data/reith/2_class_MTF_freq_experiment/time.txt', 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    print("sleeping for a bit..")
    time.sleep(2*3600)
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_0', deeper_pls=False, NetClass=NotPretrainedResnet)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_1', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=1)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_2', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=2)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_3', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=3)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_4', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=4)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################

'''
