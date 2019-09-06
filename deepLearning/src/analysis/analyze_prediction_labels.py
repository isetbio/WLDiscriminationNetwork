from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_dprime
import numpy as np
import pickle

fpath_nn = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\rule_45_on_harmonic_freq_1\1_samplesPerClass_freq_1_contrast_0_000019952623_nn_pred_labels.p'
fpath_oo = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\rule_45_on_harmonic_freq_1\1_samplesPerClass_freq_1_contrast_0_000019952623_oo_pred_label.p'

with open(fpath_oo, 'rb') as f:
    oo = pickle.load(f)
with open(fpath_nn, 'rb') as f:
    nn = pickle.load(f)

calculate_dprime(nn)
calculate_dprime(oo)