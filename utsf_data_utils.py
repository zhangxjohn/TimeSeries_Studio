import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataUtil(object):
    # Beijing PM2.5 and GefCom Electricity Price 

    # This class contains data specific information.
    # It does the following:
    #  - Read data from file
    #  - Normalise it
    #  - Split it into train, dev (validation) and test
    #  - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
    #    Every sample (x, y) shall be created as follows:
    #     - x --> window number of values
    #     - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
    #    This way X and Y will have the following dimensions:
    #     - X [number of samples, window, number of un/multivariate time series]
    #     - Y [number of samples, number of un/multivariate time series]
    
    def __init__(self, config):
        self.config = config

        self.train_rate  = self.config.train_rate
        self.valid_rate   = self.config.valid_rate
        self.w           = self.config.window
        self.h           = self.config.horizon
        self.rowdata     = self.load_dataset()
        self.n, self.m   = self.rowdata.shape            

        self.data, self.scaler = self.normalize_data(self.rowdata)

    def normalize_data(self, data, SACLED_FEATURE_RANGE=(0, 1)):

        scaler = MinMaxScaler(feature_range=SACLED_FEATURE_RANGE)
        data = scaler.fit_transform(data)

        return data, scaler

    def inverse_transform(self, y_scaled):

        y_scaler = self.scaler
        real_y = y_scaler.inverse_transform(y_scaled)

        return real_y

    def split_data(self):
    
        train_range = range(self.w + self.h - 1, int(self.train_rate * self.n))
        valid_range = range(int(self.train_rate * self.n), int((self.train_rate + self.valid_rate) * self.n))
        test_range  = range(int((self.train_rate + self.valid_rate) * self.n), self.n)
        
        train = self.get_data(train_range)
        valid = self.get_data(valid_range)
        test  = self.get_data(test_range)

        return train, valid, test
        
    def get_data(self, rng):
        n = len(rng)
        
        X = np.zeros((n, self.w, self.m))
        Y = np.zeros((n, 1))
        
        for i in range(n):
            end   = rng[i] - self.h + 1
            start = end - self.w
            
            X[i,:,:] = self.data[start : end, :]
            Y[i,:]   = self.data[rng[i], -1]
        
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

class BJPMDataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - PJPM2_5 ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape))        

    def load_dataset(self):

        df = pd.read_csv(self.data_filename, usecols=[5,6,7,8,9,10,11,12], skiprows=range(1, 25))
        df.dropna(inplace=True)
        df['cbwd'] = LabelEncoder().fit_transform(df['cbwd'])
        pm2_5 = df['pm2.5']
        df.drop(labels=['pm2.5'], axis=1, inplace=True)
        df.insert(len(df.columns), 'pm2.5', pm2_5)

        rowdata = df.values.astype('float32')

        return rowdata


class GefComPriceDataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - GefCom2014_Price ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata        


class PollutionDataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - Pollution ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata  


class BikeDataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - Bike ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata 


class NSW2013Dataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - NSW2013 ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata 


class NSW2016Dataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - NSW2016 ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata 


class TAS2016Dataset(DataUtil):

    def __init__(self, data_filename, config):
        self.data_filename = data_filename
        DataUtil.__init__(self, config)

        print('Prepare DataSet - TAS2016 ...')
        self.train, self.valid, self.test = self.split_data()
        print('[Train Set] | X-shape: {}  ---  y-shape: {}'.format(self.train[0].shape, self.train[1].shape))
        print('[Valid Set] | X-shape: {}  ---  y-shape: {}'.format(self.valid[0].shape, self.valid[1].shape))
        print('[Test  Set] | X-shape: {}  ---  y-shape: {}'.format(self.test[0].shape, self.test[1].shape)) 

    def load_dataset(self):

        df = pd.read_csv(self.data_filename)
        df.dropna(inplace=True)

        rowdata = df.values.astype('float32')

        return rowdata 





