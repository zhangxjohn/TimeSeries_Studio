import os, gc, time, datetime, argparse

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models.rnn_models import naive_RNNs, LSTMs, GRUs
from models.fcn import FCN
from models.lstm_fcn import LSTM_FCN, ALSTM_FCN
from models.lstnet import LSTNet, AR
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], \
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

keras.backend.clear_session()
absolute_path = '/mnt/nfsroot/zhangxj/ts_univariate_forecasting'

Data_Config = {
    'Beijing_PM2_5': BJPMConfig(),
    # 'GefCom2014_Price': GefComPriceConfig(),
    # 'exchange_rate': EXchange_RateConfig(),
}

Model = {
    # 'naive_RNN': naive_RNN, 
    # 'LSTM': LSTMs,
    # 'GRU': GRUs,
    'LSTNet': LSTNet,
    # 'LSTM_FCN': LSTM_FCN,
    # 'ALSTM_FCN': ALSTM_FCN,

    # 'FCN': FCN,
    # 'TCN': TCN,
    # 'ResNet': ResNet,
    # 'Transformer': Transformer,
    # 'AR': AR
}

parser = argparse.ArgumentParser('Univariate time series forecasting')
parser.add_argument('--task', type=str, default='multivariate_to_univariate', \
            help='forecasting task pattern: univariate_to_univariate or multivariate_to_univariate')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--window_and_timesteps', type=int, default=24, help='window size, i.e., time steps')
parser.add_argument('--CNNFilters', type=int, default=32, help='number of CNN hidden units')
parser.add_argument('--RNNUnits', type=int, default=32, help='number of RNN hidden units')
parser.add_argument('--RNNSkipUnits', type=int, default=10)
parser.add_argument('--skip', type=int, default=7, help='period')
parser.add_argument('--highway_window', type=int, default=7, help='The window size of the highway component')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout some neural net units for robust')
parser.add_argument('--optim', type=str, default='Adam', help='optional: SGD, RMSprop, and Adam')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--batch_size', type=int, default=128, help='network update batch size')
parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--loss', type=str, default='mae', help='loss function to use. Default=mean_absolute_error')
args = parser.parse_args()


def main(net_init, config, data, data_name, MODEL, model_name, file_res):
    X_train, y_train = data.get_dataset(ds_type='Train')
    X_valid, y_valid = data.get_dataset(ds_type='Valid')
    X_test, y_test = data.get_dataset(ds_type='Test')
    
    if args.task == 'univariate_to_univariate':
        X_train, X_valid, X_test = X_train[:, :, -1], X_valid[:, :, -1], X_test[:, :, -1]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))        
    
    model = MODEL(net_init, input_shape=X_train.shape[1:])

    if args.optim =='SGD':
        optimizer = keras.optimizers.SGD(learning_rate=config.lr, momentum=0.9, decay=0.01, nesterov=False)
    elif args.optim == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=config.lr, rho=0.9, decay=0.01, epsilon=None)
    elif args.optim == 'Adam': 
        optimizer = keras.optimizers.Adam(learning_rate=config.lr, clipnorm=args.clip)

    model.compile(loss=args.loss, optimizer=optimizer, metrics=config.metrics)
    model.summary()
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=80, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=100)
    ]

    # Training Model
    train(model, X_train, y_train, callbacks, X_valid, y_valid)

    # Testing Evaluation
    temp_arr = np.zeros((X_test.shape[0], config.D-1))
    pred_test = model.predict(X_test)
    pred_test = data.scaler.inverse_transform(np.concatenate((temp_arr, pred_test), axis=1))[:, -1].reshape((-1, 1))
    y_test = data.scaler.inverse_transform(np.concatenate((temp_arr, y_test), axis=1))[:, -1].reshape((-1, 1))
    test_mae = Mean_Absolute_Error(y_test, pred_test)
    test_rmse = Root_Mean_Squared_Error(y_test, pred_test)
    print('The current dataset:', data_name, '  |  The current model:', model_name)
    print('[Test - horizon: {}]  MAE - {:.4f}      RMSE - {:.4f}'.format(config.horizon, test_mae, test_rmse))      
    file_res.write('{}, {}, {}, {}, {}, {}\n'.format(datetime.datetime.now(), config.horizon, net_init.RNNUnits, net_init.CNNFilters, test_mae, test_rmse))

    del model, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test
    gc.collect()


def train(model, X_train, y_train, callbacks, X_valid=None, y_valid=None):
    val_data = (X_valid, y_valid) if len(X_valid)!=0 else None
    
    start_time = time.time()

    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=val_data,callbacks=callbacks)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Training. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    net_init = NetInit()

    for data_name, config in zip(Data_Config.keys(), Data_Config.values()):
        for model_name, model in zip(Model.keys(), Model.values()):
            print('The current dataset:', data_name, '  |  The current model:', model_name)
            file_res = open(f'/home/zhangxj/program/TimeSeries/results/utsf/{data_name}_{args.task}_{model_name}_res.csv', 'a+')
            file_res.write('moment, horizon, rnn-units, cnn-units, MAE, RMSE\n')

            # for horizon in [3, 6, 12, 24]:
            #     config.horizon = horizon
            #     DataUtil = BJPMDataset if data_name == 'Beijing_PM2_5' else GefComPriceDataset
            #     data = DataUtil(data_filename=f'{absolute_path}/{args.data_name}.csv', config=config)
            #     for rnn_nb in [32]:
            #         for cnn_nb in [32]:
            #             net_init.RNNUnits = rnn_nb
            #             net_init.CNNFilters = cnn_nb
            #             net_init.FeatDims = config.K
            #             net_init.time_steps = config.window

            #             main(net_init, config, data, data_name, model, model_name, file_res)                        

            config.horizon = args.horizon
            net_init.FeatDims = config.K
            net_init.RNNUnits = args.RNNUnits
            net_init.SkipRNNUnits = args.RNNSkipUnits
            net_init.CNNFilters = args.CNNFilters
            net_init.dropout = args.dropout
            net_init.skip = args.skip
            net_init.highway_window = args.highway_window
            net_init.time_steps = config.window = args.window_and_timesteps
            DataUtil = BJPMDataset if data_name == 'Beijing_PM2_5' else GefComPriceDataset
            data = DataUtil(data_filename=f'{absolute_path}/{data_name}.csv', config=config)
            main(net_init, config, data, data_name, model, model_name, file_res)

            file_res.close()

