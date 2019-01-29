from sklearn import svm
import numpy as np
from deepLearning.src.data.mat_data import get_h5mean_data, poisson_noise_loader
from deepLearning.src.models.optimal_observer import calculate_dprime
import os
import csv
import datetime
import time


def write_svm_csv(acc, dprime, metric, out_path, lock=None, metric_name='contrast'):
    if lock is not None:
        lock.acquire()
    svm_csv = os.path.join(out_path, "svm_results.csv")
    file_exists = os.path.isfile(svm_csv)

    with open(svm_csv, 'a') as csv_file:
            headers = ['svm_accuracy', 'dprime_accuracy', metric_name]
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'svm_accuracy': acc, 'dprime_accuracy': dprime, metric_name: metric})
    if lock is not None:
        lock.release()


def get_svm_accuracy(path_mat, num_samples=15000, **kwargs):
    start = time.time()
    meanData, meanDataLabels, dataMetric = get_h5mean_data(path_mat, **kwargs)
    testDataFull, testLabelsFull = poisson_noise_loader(meanData, size=num_samples, numpyData=True)
    testDataFull = testDataFull.reshape(testDataFull.shape[0], -1)
    svc = svm.LinearSVC()
    num_data = len(testDataFull)
    num_train = int(num_data*0.8)
    x_train, y_train = testDataFull[:num_train], testLabelsFull[:num_train]
    x_test, y_test = testDataFull[num_train:], testLabelsFull[num_train:]
    svc.fit(x_train, y_train)
    preds = svc.predict(x_test)
    acc = np.mean(preds == y_test)
    dprime = calculate_dprime(np.stack([preds, y_test], axis=1))
    print(f'Accuracy is {acc}, Dprime is {dprime} num train samples is {num_train}, took {str(datetime.timedelta(seconds=time.time()-start))}.')
    return acc, dprime, float(dataMetric[0])


if __name__ == '__main__':
    print("starting out..")
    path_mat = '/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_1/5_samplesPerClass_freq_1_contrast_oo_0_000414616956.h5'
    sample_numbers = np.logspace(np.log10(500), np.log10(50000), num=15).astype(np.int)
    for num in sample_numbers:
        get_svm_accuracy(path_mat, num_samples=num, includeShift=True)
