import numpy as np


class DataUtil(object):
    """
        Electricity | EXchange_Rate | Solar_AL | Traffic

        This class contains data specific information.
        It does the following:
        - Read data from file
        - Normalise it
        - Split it into train, dev (validation) and test
        - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
        Every sample (x, y) shall be created as follows:
            - x --> window number of values
            - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
        This way X and Y will have the following dimensions:
            - X [number of samples, window, number of multivariate time series]
            - Y [number of samples, number of multivariate time series]
    """

    def __init__(self, data_filename, config):
        self.config = config

        print('Prepare DataSet ...')
        with open(data_filename, 'r') as filename:
            self.rawdata   = np.loadtxt(filename, delimiter=',')

        self.w         = self.config.window
        self.h         = self.config.horizon
        self.normalize = self.config.normalize
        self.n, self.m = self.rawdata.shape
        self.scale     = np.ones(self.m)
        self.data      = np.zeros(self.rawdata.shape)

        self.normalize_data(self.normalize)
        
        self.split_data(self.config.train_rate, self.config.valid_rate)

        
    def normalize_data(self, normalize):

        if normalize == 0: # do not normalize
            self.data = self.rawdata
        
        if normalize == 1: # same normalization for all timeseries
            self.data = self.rawdata / np.max(self.rawdata)
            for i in range(self.m):
                self.scale[i] =  np.max(self.rawdata)
        
        if normalize == 2: # normalize each timeseries alone. This is the default mode
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdata[:, i]))+1
                self.data[:, i] = self.rawdata[:, i] / self.scale[i]

    def inverse_transform(self, y_scaled):
        y_real = y_scaled.copy()
        for i in range(self.m):
            y_real[:, i] = y_scaled[:, i] * self.scale[i]
        return y_real

    def split_data(self, train_rate, valid_rate):

        train_range = range(self.w + self.h - 1, int(train_rate * self.n))
        valid_range = range(int(train_rate * self.n), int((train_rate + valid_rate) * self.n))
        test_range  = range(int((train_rate + valid_rate) * self.n), self.n)
        
        self.train = self.get_data(train_range)
        self.valid = self.get_data(valid_range)
        self.test  = self.get_data(test_range)
        
    def get_data(self, rng):
        n = len(rng)
        
        X = np.zeros((n, self.w, self.m))
        Y = np.zeros((n, self.m))
        
        for i in range(n):
            end   = rng[i] - self.h + 1
            start = end - self.w
            
            X[i, :, :] = self.data[start:end, :]
            Y[i, :]   = self.data[rng[i], :]
        
        return [X, Y]

    def get_dataset(self, ds_type='Train'):
        if ds_type == 'Train':
            ds = self.train
        elif ds_type == 'Valid':
            ds = self.valid
        elif ds_type == 'Test':
            ds = self.test
        else:
            raise RuntimeError("Unknown dataset type['Train', 'Valid', 'Test']:", ds_type)            

        return ds[0], ds[1]





class M2UDataUtil(object):
    """
        Electricity | EXchange_Rate | Solar_AL | Traffic

        This class contains data specific information.
        It does the following:
        - Read data from file
        - Normalise it
        - Split it into train, dev (validation) and test
        - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
        Every sample (x, y) shall be created as follows:
            - x --> window number of values
            - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
        This way X and Y will have the following dimensions:
            - X [number of samples, window, number of multivariate time series]
            - Y [number of samples, number of multivariate time series]
    """

    def __init__(self, data_filename, config):
        self.config = config

        print('Prepare DataSet ...')
        with open(data_filename, 'r') as filename:
            self.rawdata   = np.loadtxt(filename, delimiter=',')

        self.pred_dim  = self.config.pred_dim
        self.w         = self.config.window
        self.h         = self.config.horizon
        self.normalize = self.config.normalize
        self.n, self.m = self.rawdata.shape
        self.scale     = np.ones(self.m)
        self.data      = np.zeros(self.rawdata.shape)

        self.normalize_data(self.normalize)
        
        self.split_data(self.config.train_rate, self.config.valid_rate)

        
    def normalize_data(self, normalize):

        if normalize == 0: # do not normalize
            self.data = self.rawdata
        
        if normalize == 1: # same normalization for all timeseries
            self.data = self.rawdata / np.max(self.rawdata)
            for i in range(self.m):
                self.scale[i] =  np.max(self.rawdata)
        
        if normalize == 2: # normalize each timeseries alone. This is the default mode
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdata[:, i]))+1
                self.data[:, i] = self.rawdata[:, i] / self.scale[i]

    def inverse_transform(self, y_scaled):
        y_real = y_scaled.copy()
        for i in range(self.m):
            y_real[:, i] = y_scaled[:, i] * self.scale[i]
        return y_real

    def split_data(self, train_rate, valid_rate):

        train_range = range(self.w + self.h - 1, int(train_rate * self.n))
        valid_range = range(int(train_rate * self.n), int((train_rate + valid_rate) * self.n))
        test_range  = range(int((train_rate + valid_rate) * self.n), self.n)
        
        self.train = self.get_data(train_range)
        self.valid = self.get_data(valid_range)
        self.test  = self.get_data(test_range)
        
    def get_data(self, rng):
        n = len(rng)
        
        X = np.zeros((n, self.w, self.m))
        Y = np.zeros((n, 1))
        
        for i in range(n):
            end   = rng[i] - self.h + 1
            start = end - self.w
            
            X[i, :, :] = self.data[start : end, :]
            Y[i, :]   = self.data[rng[i], self.pred_dim]
        
        return [X, Y]

    def get_dataset(self, ds_type='Train'):
        if ds_type == 'Train':
            ds = self.train
        elif ds_type == 'Valid':
            ds = self.valid
        elif ds_type == 'Test':
            ds = self.test
        else:
            raise RuntimeError("Unknown dataset type['Train', 'Valid', 'Test']:", ds_type)            

        return ds[0], ds[1]
