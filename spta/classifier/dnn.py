from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from spta.util import log as log_util


class ClassifierDNNParams(namedtuple('ClassifierDNNParams', 'name model_path labels_path window_size')):
    '''
    Parameters for ClassifierDNNParams model
    '''
    __slots__ = ()

    def __repr__(self):
        '''
        Override the representation of MeanOfPastParams
        # https://stackoverflow.com/a/7914212/3175179
        '''
        as_str = 'classifier-{}-{}'
        return as_str.format(self.name, self.window_size)

    def __str__(self):
        return super(ClassifierDNNParams, self).__repr__()


class ClassifierDNN(log_util.LoggerMixin):
    '''
    A classifier based on a deep neural network, that has been developed externally and saved as a h5py object
    '''

    def __init__(self, classifier_params):
        # use the model params to open the h5py object
        # model_lstm_wSW_label_k-seed-ci.h5

        self.classifier_params = classifier_params

        # load the CSV of labels
        self.label_names = pd.read_csv(self.classifier_params.labels_path)
        num_labels = self.label_names.shape[0]

        # we need to rebuild the model...
        # TODO manage different models
        model = Sequential()
        model.add(LSTM(classifier_params.window_size, return_sequences=True, input_shape=(classifier_params.window_size, 1)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(100))
        model.add(Dense(num_labels, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=0.0001)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        print(model.summary())

        # Load weights
        model.load_weights(classifier_params.model_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    def label_for_series(self, series_of_len_tp):

        window_size = self.classifier_params.window_size

        # normalize series, need to reshape it first
        s = np.array(series_of_len_tp).reshape(1, window_size)
        self.logger.debug('About to normalize: {}'.format(s))
        normalized_series = normalize(s, norm='l1')
        self.logger.debug('Normalized to: {}'.format(normalized_series))

        # transform series to model input
        input_series = np.array(normalized_series).reshape(1, window_size, 1)

        # predict the result to get an encoded label
        result = self.model.predict(input_series)
        label_predict = np.argmax(result)

        # find the label from the CSV
        label_solver = self.label_names.iloc[label_predict, 1]
        self.logger.debug('Got label: {}'.format(label_solver))

        return label_solver
        # return '2-0-1'
