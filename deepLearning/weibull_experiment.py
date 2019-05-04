from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from psychopy.data import FitWeibull


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


class FixedBetaFitWeibulli(FitWeibull):
    def _eval(xx, alpha):
        _chance = 0
        xx = np.asarray(xx)
        yy = _chance + (1.0 - _chance) * (1 - np.exp(-(xx / alpha) ** fixed_beta))
        return yy

class FixedBetaFitWeibull(FitWeibull):
    """
    Fits the weibull cmf given a fixed beta.

    should probably have the _baseFunctionFit class as superclass, but FitWeibull is easier to import and
    essentially the same..
    """
    glob_beta = None
    def __init__(self, fixed_beta, *args, **kwargs):
        self.fixed_beta = fixed_beta
        FixedBetaFitWeibull.glob_beta = self.fixed_beta
        super().__init__(*args, **kwargs)

    def _eval(self, xx, alpha):
        _chance = 0
        xx = np.asarray(xx)
        yy = _chance + (1.0 - _chance) * (1 - np.exp(-(xx / alpha) ** self.fixed_beta))
        return yy

    def _inverse(self, yy, alpha):
        _chance = 0
        xx = alpha * (-np.log((1.0 - yy)/(1 - _chance))) ** (1.0/self.fixed_beta)
        return xx


def get_alphas_fixed_beta(x, *ys):
    """
    We can expect that y1.max() == y2.max()
    svm dprime max is sometimes smaller
    :param x:
    :param y1:
    :param y2:
    :return:
    """
    fits = []
    for i in range(len(ys)):
        ys[i][ys[i] == ys[i].max()] = np.max(ys)
        fits.append(FitWeibull(x, ys[i]/ys[i].max(), expectedMin=0))
        print(f"fit{i} is {fits[i].params}.")

    fixed_beta = np.mean([fit.params[1] for fit in fits])
    print(f"fixed_beta = {fixed_beta}")
    new_fits = []
    for i, y in enumerate(ys):
        new_fit = FixedBetaFitWeibull(fixed_beta, x, y/y.max(), expectedMin=0)
        new_fits.append(new_fit)
        print(f"new_fit{i} is {new_fits[i].params}")
    alphas = [new_fit.params[0] for new_fit in new_fits]
    return alphas, new_fits


folder = '/share/wandell/data/reith/redo_experiments/sensor_harmonic_rotation/'
metric = 'angle'
csv1 = os.path.join(folder, 'results.csv')
csv_svm = os.path.join(folder, 'svm_results.csv')
oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
contrasts = get_csv_column(csv1, metric, sort_by=metric)
fitter = FitWeibull(contrasts, oo / oo.max(), expectedMin=0)
get_alphas_fixed_beta(contrasts, oo, nn)
print(fitter.params)
plt.plot(contrasts, oo)
plt.plot(contrasts, fitter.eval(contrasts) * oo.max())
print('nice')
