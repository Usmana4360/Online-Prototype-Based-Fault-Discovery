import numpy as np, pandas as pd
errors = pd.read_csv('results/global_reconstruction_error.csv')
labels = np.load('data/labels/test_labels.npy')
print('Error windows :', len(errors))
print('Label windows :', len(labels))
print('Match         :', len(errors) == len(labels))