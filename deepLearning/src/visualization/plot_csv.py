import csv
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')
from scipy.interpolate import spline, BSpline













if __name__ == '__main__':
    csv_path = "/share/wandell/data/reith/matlabData/shift_contrast100/results.csv"
    shiftVals = "/share/wandell/data/reith/matlabData/shift_contrast100/shiftVals.txt"
    smooth = False
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        resnet_accuracy = []
        optimal_observer_accuracy = []
        theoretical_d_index = []
        contrast_values = []
        shift_values = []
        for row in reader:
            optimal_observer_accuracy.append(float(row['optimal_observer_accuracy']))
            resnet_accuracy.append(float(row['ResNet_accuracy']))
            theoretical_d_index.append(float(row['theoretical_d_index']))
            contrast_values.append(float(row['contrast']))
            shift_values.append(float(row['shift']))
            # print(row)
        resnet_accuracy = np.array(resnet_accuracy)
        theoretical_d_index = np.array(theoretical_d_index)
        contrast_values = np.array(contrast_values)
        optimal_observer_accuracy = np.array(optimal_observer_accuracy)
        shift_values = np.array(shift_values)
    with open(shiftVals, 'r') as f:
        shift_values = f.read().splitlines()
        shift_values = np.array(shift_values).astype(np.float)
    sort_indices = np.argsort(shift_values)
    contrast_values = contrast_values[sort_indices]
    resnet_accuracy = resnet_accuracy[sort_indices]
    shift_values = shift_values[sort_indices]
    originalPhase = 1.570796326794896558
    shift_values = shift_values - originalPhase
    degrees = shift_values*1500
    seconds = degrees/3600
    optimal_observer_accuracy = optimal_observer_accuracy[sort_indices]
    fig = plt.figure()
    ax = plt.axes()
    plt.xscale('log')
    plt.xlabel('Signal shift in sec steps')
    plt.ylabel('Accuracy')
    plt.title('Freq 1 (signal/signal+shift) accuracy for various shift values')
    if smooth:
        x_new = np.linspace(contrast_values.min(), contrast_values.max(), 200)
        spl = BSpline()
        resnet_smooth = spline(contrast_values, resnet_accuracy, x_new)
        optimal_observer_smooth = spline(contrast_values, optimal_observer_accuracy, x_new)
        plt.plot(x_new, optimal_observer_smooth, label='Optimal Observer Accuracy')
        plt.plot(x_new, resnet_smooth, label='ResNet Accuracy')
    else:
        plt.plot(seconds, optimal_observer_accuracy, label='Optimal Observer Accuracy')
        plt.plot(seconds, resnet_accuracy, label='ResNet Accuracy')
    plt.legend(frameon=True)

    fig.show()
    print('done!')



