import os
import numpy as np
from scipy.io import loadmat


def ReadData(mat_file):
    dataset = os.path.basename(mat_file).rstrip('.mat')
    print('Processing dataset: {}'.format(dataset))

    mat = loadmat(mat_file)
    print(mat.keys())

    X_train_mat, y_train_mat, X_test_mat, y_test_mat = np.squeeze(mat['X_train']), \
                                                       np.squeeze(mat['y_train']), np.squeeze(
        mat['X_test']), np.squeeze(mat['y_test'])

    assert len(X_train_mat) == len(y_train_mat)
    assert len(X_test_mat) == len(y_test_mat)

    y_train = y_train_mat.reshape(-1, 1)
    y_test = y_test_mat.reshape(-1, 1)

    var_list = []
    for i in range(X_train_mat.shape[0]):
        var_count = X_train_mat[i].shape[0]
        var_list.append(var_count)

    var_list = np.array(var_list)
    max_nb_timesteps = var_list.max()

    X_train = np.zeros((X_train_mat.shape[0], max_nb_timesteps, X_train_mat[0].shape[-1]))
    X_test = np.zeros((X_test_mat.shape[0], max_nb_timesteps, X_test_mat[0].shape[-1]))

    # pad ending with zeros to get numpy arrays
    for i in range(X_train_mat.shape[0]):
        var_count = X_train_mat[i].shape[0]
        X_train[i, :var_count, :] = X_train_mat[i]

    for i in range(X_test_mat.shape[0]):
        var_count = X_test_mat[i].shape[0]
        X_test[i, :var_count, :] = X_test_mat[i][:max_nb_timesteps, :]

    nb_classes = len(np.unique(y_train))

    # X_train and X_test's shape: (nb_samples, time_steps, input_dim)
    # y_train and y_test's shape: (nb_sample, 1)
    return X_train, y_train, X_test, y_test, nb_classes


