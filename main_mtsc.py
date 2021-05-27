import os, gc, time, datetime, argparse
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
from mtsc_data_utils import ReadData
from eval_metrics import Accuracy_Score, Precision_Score, Recall_Score, Auc_Score, F1_Score

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], \
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

tf.random.set_seed(2021)
keras.backend.clear_session()

Data_Config = {
    'ArabicDigits': 10,
    'AUSLAN': 95,
    'CharacterTrajectories': 20,
    'CMUsubject16': 2,
    'DigitShapes': 4,
    'ECG': 2,
    'JapaneseVowels': 9,
    'KickvsPunch': 2,
    'LIBRAS': 15,
    'NetFlow': 2,
    'PEMS': 7,
    'PenDigits': 10,
    'RobotFailureLP1': 4,
    'RobotFailureLP2': 5,
    'RobotFailureLP3': 4,
    'RobotFailureLP4': 3,
    'RobotFailureLP5': 5,
    'Shapes': 3,
    'UWave': 8,
    'Wafer': 2,
    'WalkvsRun': 2
}

Model = {
    'naive_RNN': naive_RNN,
    'LSTM': LSTM,
    'GRU': GRU,
    'FCN': FCN,
    'LSTM_FCN': LSTM_FCN,
    'ALSTM_FCN': ALSTM_FCN,
    'ResNet': ResNet,
    'TCN': TCN,
    'LSTNet': LSTNet,
    'Transformer': Transformer
}

parser = argparse.ArgumentParser('Multivariate time series forecasting')
parser.add_argument('--optim', type=str, default='Adam', help='optional: SGD, RMSprop, and Adam')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='network update batch size')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--loss', type=str, default='categorical_crossentropy',
                    help='loss function to use. Default=mean_absolute_error')
args = parser.parse_args()


def train(model, X_train, y_train, callbacks, X_valid=None, y_valid=None):
    start_time = time.time()
    if X_valid.any() != None:
        val_data = (X_valid, y_valid)
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=val_data, callbacks=callbacks)
    else:
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, callbacks=callbacks)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Training. Total elapsed time (h:m:s): {}".format(elapsed))


def main(net_init, data_path, data_classes, data_name, MODEL, model_name, file_res):
    X_train, y_train, X_test, y_test, nb_classes = ReadData(data_path)
    assert data_classes == nb_classes

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    args.batch_size = min(int(X_train.shape[0]/10), 16)
    model = MODEL(net_init, input_shape=X_train.shape[1:])

    if args.optim == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=args.learing_rate, momentum=0.0, decay=0.0, nesterov=False)
    elif args.optim == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=args.learing_rate, rho=0.9, epsilon=1e-8, decay=0.0)
    else:  # Adam
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)

    model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=60)
    ]

    # Training Model
    train(model, X_train, y_train, callbacks, X_valid=X_test, y_valid=y_test)

    # Testing Evaluation
    pred_test = model.predict(X_test)
    test_auc = Auc_Score(y_test, pred_test)
    pred_test = np.argmax(pred_test, axis=1)
    y_test = np.argmax(y_test, axis=1)
    test_acc = Accuracy_Score(y_test, pred_test)
    test_precision = Precision_Score(y_test, pred_test)
    test_recall = Recall_Score(y_test, pred_test)
    test_f1 = F1_Score(y_test, pred_test)
    print('The current dataset:', data_name, '  |  The current model:', model_name)
    print('[Test]  Accuracy - {:.4f}   Precision - {:.4f}   Recall - {:.4f}   AUC - {:.4f}   F1 - {:.4f}'
                .format(test_acc, test_precision, test_recall, test_auc, test_f1))
    file_res.write('{}, {}, {}, {}, {}, {}, {}\n'.format(datetime.datetime.now(), net_init.RNNUnits, test_acc, test_precision, test_recall, test_auc, test_f1))

    del X_train, y_train, X_test, y_test, model, optimizer
    gc.collect()


if __name__ == '__main__':
    net_init = NetInit()

    for data_name, data_classes in zip(Data_Config.keys(), Data_Config.values()):
        data_path = '/mnt/nfsroot/zhangxj/ts_multivariate_classification//TSDatasets/' + data_name + '.mat'
        for model_name, model in zip(Model.keys(), Model.values()):
            print('The current dataset:', data_name, '  |  The current model:', model_name)

            file_res = open(f'/home/zhangxj/program/results/mtsc/{data_name}_{model_name}_results.csv', 'a+')
            file_res.write('moment, rnn_units, Accuracy, Precision, Recall, AUC, F1\n')

            for n in [16, 32, 64]:
                net_init.RNNUnits = n
                net_init.FeatDims = data_classes

                main(net_init, data_path, data_classes, data_name, model, model_name, file_res)

            file_res.close()
