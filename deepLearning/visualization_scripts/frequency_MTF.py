import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import bisect


def get_csv_column(csv_path, col_name, sort_by=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    return col


MTF_path = '/share/wandell/data/reith/2_class_MTF_freq_experiment/'
fname = 'Modular_transfer_function_frequencies'
if os.path.exists(os.path.join(MTF_path, f'{fname}.png')):
    os.remove(os.path.join(MTF_path, f'{fname}.png'))
target_d = 2

frequency_paths = [os.path.join(MTF_path, freq_dir) for freq_dir in os.listdir(MTF_path)]

freqs =[]
nn_dprimes = []
oo_dprimes = []

for p in frequency_paths:
    if os.path.isdir(p):
        freq = int(p.split('_')[-1])
        freqs.append(freq)
        nn_dprimes.append(get_csv_column(os.path.join(p, 'results.csv'), 'nn_dprime', sort_by='contrast'))
        oo_dprimes.append(get_csv_column(os.path.join(p, 'results.csv'), 'optimal_observer_d_index', sort_by='contrast'))

sort_idxs = np.argsort(freqs)
freqs, nn_dprimes, oo_dprimes = np.array(freqs), np.array(nn_dprimes), np.array(oo_dprimes)
freqs = freqs[sort_idxs]
nn_dprimes = nn_dprimes[sort_idxs]
oo_dprimes = oo_dprimes[sort_idxs]

contrasts = get_csv_column(os.path.join(frequency_paths[0], 'results.csv'), 'contrast', sort_by='contrast')
nn_bilinear_targets = []
oo_bilinear_targets = []

for dprimes in nn_dprimes:
    right_target = bisect.bisect(dprimes, target_d)
    left_target = right_target -1
    p_val = (target_d - dprimes[left_target])/(dprimes[right_target]-dprimes[left_target])
    interpolated_val = (1-p_val) * contrasts[left_target] + p_val * contrasts[right_target]
    nn_bilinear_targets.append(interpolated_val)

for dprimes in oo_dprimes:
    right_target = bisect.bisect(dprimes, target_d)
    left_target = right_target -1
    p_val = (target_d - dprimes[left_target])/(dprimes[right_target]-dprimes[left_target])
    print(p_val, contrasts[left_target])
    interpolated_val = (1-p_val) * contrasts[left_target] + p_val * contrasts[right_target]
    oo_bilinear_targets.append(interpolated_val)

nn_bilinear_targets = np.array(nn_bilinear_targets)
oo_bilinear_targets = np.array(oo_bilinear_targets)

fig = plt.figure()
plt.xlabel('frequency')
plt.ylabel('contrast needed for a dprime of 2')
plt.title('Modular Transfer Function - Harmonic curve with various frequencies')

plt.plot(freqs, nn_bilinear_targets, label='ResNet')
plt.plot(freqs, oo_bilinear_targets, label='Optimal Observer')
plt.legend(frameon=True)

fig.savefig(os.path.join(MTF_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
