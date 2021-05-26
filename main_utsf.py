import os, time, datetime, argparse

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models.rnn_models import naive_RNN, LSTM, GRU
from models.fcn import FCN
from models.lstm_fcn import LSTM_FCN, ALSTM_FCN
from models.lstnet import LSTNet
from models.resnet import ResNet
from models.tcn import TCN
from models.transformer import Transformer

# from models.mtnet import MTNet
# from models.darnn import DARNN
# from models.nbeat import NbeatsNet

from net_init_utils import NetInit
from configs import BJPMConfig, GefComPriceConfig
from utsf_data_utils import BJPMDataset, GefComPriceDataset
from eval_metrics import Mean_Absolute_Error, Root_Mean_Squared_Error

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], \
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

keras.backend.clear_session()

Model = GRU
Config = BJPMConfig
Dataset = BJPMDataset

parser = argparse.ArgumentParser('Univariate time series forecasting')
parser.add_argument('--model', type=str, default='GRU')
parser.add_argument('--data_name', type=str, default='Beijing_PM2_5', help='data file name')
parser.add_argument('--task', type=str, default='multivariate_to_univariate', \
            help='forecasting task pattern: univariate_to_univariate or multivariate_to_univariate')
parser.add_argument('--optim', type=str, default='Adam', help='optional: SGD, RMSprop, and Adam')
parser.add_argument('--batch_size', type=int, default=128, help='network update batch size')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--loss', type=str, default='mae', help='loss function to use')
args = parser.parse_args()


def train(model, X_train, y_train, callbacks, X_valid=None, y_valid=None):
    if len(X_valid)!=0:
        val_data = (X_valid, y_valid)
    else:
        val_data = None
    
    start_time = time.time()

    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=val_data,callbacks=callbacks)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Training. Total elapsed time (h:m:s): {}".format(elapsed))

    return history


if __name__ == '__main__':
    absolute_path = '/mnt/nfsroot/zhangxj/ts_univariate_forecasting'
    config = Config()
    net_init = NetInit()

    # f = open(f'/home/zhangxj/program/results/utsf/{args.data_name}_{args.task}_{args.model}_utsf_testing.csv', 'a+')
    # f.write('horizon, units, MAE, RMSE\n')

    for horizon in [3]:
        config.horizon = horizon
        for rnn_units in [32]:
            net_init.RNNUnits = rnn_units
            net_init.FeatDims = config.K

            data = Dataset(data_filename=f'{absolute_path}/{args.data_name}.csv', config=config)

            X_train, y_train = data.get_dataset(ds_type='Train')
            X_valid, y_valid = data.get_dataset(ds_type='Valid')
            X_test, y_test = data.get_dataset(ds_type='Test')
            
            if args.task == 'univariate_to_univariate':
                X_train, X_valid, X_test = X_train[:, :, -1], X_valid[:, :, -1], X_test[:, :, -1]
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))        
            
            model = Model(net_init, input_shape=X_train.shape[1:])

            if args.optim =='SGD':
                optimizer = keras.optimizers.SGD(learning_rate=config.lr, momentum=0.0, decay=0.0, nesterov=False)
            elif args.optim == 'RMSprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=config.lr, rho=0.9, decay=0.0, epsilon=None)
            elif args.optim == 'Adam': 
                optimizer = keras.optimizers.Adam(learning_rate=config.lr, clipnorm=args.clip)

            model.compile(loss=args.loss,
                          optimizer=optimizer,
                          metrics=config.metrics)

            callbacks = [
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=80, min_lr=0.0001),
                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=100)
            ]

            # Training Model
            hist = train(model, X_train, y_train, callbacks, X_valid, y_valid)

            log = pd.DataFrame(hist.history)
            print('The min loss:', log.loc[log['loss'].idxmin]['loss'], 'The MAE under the min loss:',  log.loc[log['loss'].idxmin]['mean_absolute_error'])

            # Testing Evaluation
            temp_arr = np.zeros((X_test.shape[0], config.D-1))
            pred_test = model.predict(X_test)
            pred_test = data.scaler.inverse_transform(np.concatenate((temp_arr, pred_test), axis=1))[:, -1].reshape((-1, 1))
            y_test = data.scaler.inverse_transform(np.concatenate((temp_arr, y_test), axis=1))[:, -1].reshape((-1, 1))
            test_mae = Mean_Absolute_Error(y_test, pred_test)
            test_rmse = Root_Mean_Squared_Error(y_test, pred_test)
            print('[Test - horizon: {} | hidden: {}]  MAE - {:.4f}      RMSE - {:.4f}'.format(h, n, test_mae, test_rmse))    
            
    #         f.write('{},{},{},{}\n'.format(h, n, test_mae, test_rmse))
    # f.close()
