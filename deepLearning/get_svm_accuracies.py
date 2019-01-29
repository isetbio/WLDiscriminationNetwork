from deepLearning.src.models.support_vector_machine import write_svm_csv, get_svm_accuracy
import os
import multiprocessing as mp
from glob import glob
import time
import datetime

def h5gen(folder):
    h5_files = glob(f"{folder}*.h5")
    for f in h5_files:
        yield f


def score_svm(h5_path, lock, num_samples=15000):
    metric = 'contrast'
    acc, metric_val = get_svm_accuracy(h5_path, num_samples)
    write_svm_csv(acc, metric_val, os.path.dirname(h5_path), lock=lock, metric_name=metric)


def run_svm_on_h5(folder, num_cpus):
    function_start = time.time()
    lock = mp.Lock()
    h5 = h5gen(folder)
    cpus = list(range(num_cpus))
    procs = {}
    while True:
        try:
            if procs == {}:
                for cpu in cpus:
                    h5_file = next(h5)
                    print(f"Svm scoring {h5_file}")
                    curr_p = mp.Process(target=score_svm, args=[h5_file, lock])
                    procs[cpu] = curr_p
                    curr_p.start()
            for cpu, proc in procs.items():
                if not proc.is_alive():
                    h5_file = next(h5)
                    print(f"Svm scoring {h5_file}")
                    curr_p = mp.Process(target=score_svm, args=[h5_file, lock])
                    procs[cpu] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in procs.values():
        proc.join()

    function_end = time.time()
    with open(os.path.join(folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    time.sleep(60)
    print("done!")


def subfolder_gen(super_folder):
    sub_folders = [os.path.join(super_folder, f) + '/' for f in os.listdir(super_folder)]
    for f in sub_folders:
        yield f


if __name__ == '__main__':
    super_folder = '/share/wandell/data/reith/2_class_MTF_freq_experiment/'
    folder_gen = subfolder_gen(super_folder)
    parallel_folders = list(range(7))
    num_cpus = 5
    processes = {}
    while True:
        try:
            if processes == {}:
                for f in parallel_folders:
                    sub_folder = next(folder_gen)
                    print(f"scoring {sub_folder}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus])
                    processes[f] = curr_p
                    curr_p.start()
            for f, proc in processes.items():
                if not proc.is_alive():
                    sub_folder = next(folder_gen)
                    print(f"scoring {sub_folder}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus])
                    processes[f] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in processes.values():
        proc.join()

    function_end = time.time()
    with open(os.path.join(super_folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    time.sleep(60)
    print("done!")

    run_svm_on_h5(sub_folders[0], 1)
    print("done")
