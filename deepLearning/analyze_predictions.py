import pickle
import numpy as np

picklePath = '/share/wandell/data/reith/matlabData/shift_contrast33/optimalOpredictionLabel.p'

ooPredictionLabel = pickle.load(open(picklePath, 'rb'))

predictions = ooPredictionLabel[:,0]
labels = ooPredictionLabel[:,1]

print(f'Accuracy is {np.mean(predictions==labels)*100}%')

