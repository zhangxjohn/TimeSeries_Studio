import tensorflow as tf
from tensorflow.keras import backend as K
# univariate time series forecasting

class BJPMConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24*7
        self.horizon          = 1

        self.T                = 43824
        self.D                = 8
        self.K                = 1

        self.metrics          = [tf.keras.metrics.MeanAbsoluteError()]
        self.lr               = 0.001

class GefComPriceConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24
        self.horizon          = 1

        self.T                = 21552
        self.D                = 9
        self.K                = 1

        self.metrics          = [tf.keras.metrics.MeanAbsoluteError()]
        self.lr               = 0.001


# multivariate time series forecasting
#Electricity | EXchange_Rate | Solar_Energy | Traffic
class ElectricityConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24*7
        self.horizon          = 24
        self.normalize        = 2

        self.T                = 26304
        self.D                = 321
        self.K                = 321

        self.metrics          = [rse, corr]
        self.lr               = 0.001


class EXchange_RateConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24*7
        self.horizon          = 24
        self.normalize        = 2

        self.T                = 7588
        self.D                = 8
        self.K                = 8

        self.metrics          = [rse, corr]
        self.lr               = 0.001


class Solar_EnergyConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24*7
        self.horizon          = 24
        self.normalize        = 2

        self.T                = 52560
        self.D                = 137
        self.K                = 137

        self.metrics          = [rse, corr]
        self.lr               = 0.001


class TrafficConfig:
    def __init__(self):
        self.train_rate       = 0.6
        self.valid_rate       = 0.2
        self.window           = 24*7
        self.horizon          = 24
        self.normalize        = 2

        self.T                = 17544
        self.D                = 862
        self.K                = 862

        self.metrics          = [rse, corr]
        self.lr               = 0.001


def rse(y_true, y_pred, epsilon=1e-8):
    num = K.sqrt(K.sum(K.square(y_true - y_pred)))
    den = K.sqrt(K.sum(K.square(y_true - K.mean(y_true))))
    return num / (den+epsilon)

def corr(y_true, y_pred, epsilon=1e-8):
    num1 = y_true - K.mean(y_true, axis=0)
    num2 = y_pred - K.mean(y_pred, axis=0)
    num  = K.mean(num1 * num2, axis=0)
    den  = K.std(y_true, axis=0) * K.std(y_pred, axis=0)
    return K.mean(num / (den+epsilon))
