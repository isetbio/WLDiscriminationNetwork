from scipy.stats import exponweib
from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
import os
from psychopy.data import FitWeibull

global fixed_beta


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


def cum_weibull(x, alpha, beta, scale):
    return scale * (1 - np.exp(-(x / alpha) * beta))


class FixedBetaFitWeibull(FitWeibull):
    @staticmethod
    def _eval(xx, alpha):
        _chance = 0
        xx = np.asarray(xx)
        yy = _chance + (1.0 - _chance) * (1 -
                                          np.exp(-(xx / alpha) ** fixed_beta))
        return yy


def get_alpha_ratio(x, y1, y2):
    """
    We can expect that y1.max() == y2.max()
    :param x:
    :param y1:
    :param y2:
    :return:
    """
    fit1 = FitWeibull(x, y1/y1.max(), expectedMin=0)
    fit2 = FitWeibull(x, y2/y2.max(), expectedMin=0)
    print(f"fit1: {fit1.params}, fit2: {fit2.params}")
    global fixed_beta
    fixed_beta = (fit1.params[1] + fit2.params[1])/2
    print(f"fixed_beta = {fixed_beta}")
    fit11 = FixedBetaFitWeibull(x, y1/y1.max(), expectedMin=0)
    fit22 = FixedBetaFitWeibull(x, y2/y2.max(), expectedMin=0)
    print(f"new alphas is {fit11.params}, {fit22.params}")


folder = '/share/wandell/data/reith/redo_experiments/sensor_harmonic_rotation/'
metric = 'angle'
csv1 = os.path.join(folder, 'results.csv')
csv_svm = os.path.join(folder, 'svm_results.csv')
oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
contrasts = get_csv_column(csv1, metric, sort_by=metric)
fitter = FitWeibull(contrasts, oo / oo.max(), expectedMin=0)
fitter2 = FitWeibull(contrasts, nn / nn.max(), expectedMin=0)
# fitter3 = FixedBetaFitWeibull(contrasts, nn / nn.max(), expectedMin=0)
fitter4 = exponweib.fit([contrasts, nn / nn.max()], floc=0, fa=1)
get_alpha_ratio(contrasts, oo, svm)
print(fitter.params)
plt.plot(contrasts, oo)
plt.plot(contrasts, fitter.eval(contrasts) * oo.max())
print('nice')
