import scipy.io as sio
import os
import numpy as np

root = './datasets/ood_data/svhn'
filename = 'test_32x32.mat'

loaded_mat = sio.loadmat(os.path.join(root, filename))

data = loaded_mat['X']
targets = loaded_mat['y']

data = np.transpose(data, (3, 0, 1, 2))

selected_data = []
selected_targets = []
count = np.zeros(11)

for i, y in enumerate(targets):
    if count[y[0]] < 1000:
        selected_data.append(data[i])
        selected_targets.append(y)
        count[y[0]] += 1

selected_data = np.array(selected_data)
selected_targets = np.array(selected_targets)

selected_data = np.transpose(selected_data, (1, 2, 3, 0))

save_mat = {'X': selected_data, 'y': selected_targets}

save_filename = 'selected_test_32x32.mat'
sio.savemat(os.path.join(root, save_filename), save_mat)
