import os

import pygam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from metrics import *
from common import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter
import statsmodels.api as sm

import tensorflow as tf
from keras.layers import *


class Model(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) or NotImplemented)

    model = None
    train_time = None
    infer_time = None

    @abc.abstractmethod
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None):
        raise NotImplementedError


class RegressorModel(Model, abc.ABC):
    pass


class ClassifierModel(Model, abc.ABC):
    pass


class NaiveRegressor(RegressorModel):
    __model_data = None

    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        t_start = perf_counter()
        self.__model_data = np.mean(training_data[targets].values)
        t_stop = perf_counter()
        self.train_time = t_stop - t_start

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        t_start = perf_counter()
        preds = np.repeat(self.__model_data, len(data))
        t_stop = perf_counter()
        self.infer_time = t_stop - t_start
        return preds

    def test(self):
        pass

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True):

        self.train(training_data, features, targets)
        preds = self.predict(testing_data)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


class LinearRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}', training_data).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True):

        self.train(training_data, features, targets)
        preds = self.predict(testing_data)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


class LassoRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data).fit_regularized()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True):

        self.train(training_data, features, targets)
        preds = self.predict(testing_data)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


class GAM(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        # TODO:
        s, f = pygam.s, pygam.f
        X = training_data[features]
        self.model = pygam.GAM(s(0) + s(1) + f(2)).fit(X, training_data[targets])

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data[features])

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True):

        self.train(training_data, features, targets)
        preds = self.predict(testing_data, features)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


class RobustRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        self.model = sm.RLM.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data,
                                         M=sm.robust.norms.HuberT()).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values


class PolynomialRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):

        poly = np.polyfit()


class NNRegressor(RegressorModel):
    @staticmethod
    def denormalized_mae(y_true, y_preds):
        return tf.abs(y_true-y_preds) * 251.01737952758302 + 92.8677986740517

    def build(self, input_shape, dense_width, dense_depth, dropout, optimizer):
        inp = Input(shape=(input_shape))
        pass

    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              ):
        # convert dfs to ndarrays
        x_train = training_data[features].iloc[int(len(training_data)*.2):].values.astype('float32')
        y_train = training_data[targets].iloc[int(len(training_data)*.2):].values.astype('float32')

        x_eval = training_data[features].iloc[:int(len(training_data)*.2)].values.astype('float32')
        y_eval = training_data[targets].iloc[:int(len(training_data)*.2)].values.astype('float32')

        print(f'train samples: {x_train.shape[0]}\ntest samples: {x_eval.shape[0]}')

        # modeling
        inp = Input(shape=(x_train.shape[-1]))
        x = Dense(128, activation='relu')(inp)
        x = Dropout(.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(.4)(x)
        x = Dense(8, activation='relu')(x)
        out = Dense(1, activation='linear')(x)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer='Adam',
                           loss=tf.keras.losses.Huber(),
                           metrics=[self.denormalized_mae])

        print(self.model.summary())

        os.makedirs('./tmp', exist_ok=True)

        checkpoint = ModelCheckpoint(f'./tmp/NN_weights_best.h5',
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', patience=50)

        hist = self.model.fit(x=x_train,
                              y=y_train,
                              validation_data=(x_eval, y_eval),
                              epochs=500,
                              batch_size=1000,
                              callbacks=[checkpoint, es])

        self.model.load_weights('./tmp/NN_weights_best.h5')

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.show()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True):

        self.denorm = denorm

        self.train(training_data, features, targets)
        preds = self.predict(testing_data, features)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values.astype('float32')
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


# ********************** CLS ************************

class NaiveClassifier(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        pass

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        pass
