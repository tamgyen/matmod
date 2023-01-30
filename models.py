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

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import split_data, normalize
from metrics import score_classification, plot_confusion_mtarix
from sklearn.svm import LinearSVC, SVC

import xgboost as xgb


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
              targets: list[str] = None,
              **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       **kwargs):
        raise NotImplementedError


class RegressorModel(Model, abc.ABC):
    pass


class ClassifierModel(Model, abc.ABC):
    def __init__(self, num_classes):
        self.num_classes = num_classes


class NaiveRegressor(RegressorModel):
    __model_data = None

    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        t_start = perf_counter()
        self.__model_data = np.mean(training_data[targets].values)
        t_stop = perf_counter()
        self.train_time = t_stop - t_start

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        t_start = perf_counter()
        preds = np.repeat(self.__model_data, len(data))
        t_stop = perf_counter()

        self.preds = preds

        self.infer_time = t_stop - t_start
        return preds

    def test(self):
        pass

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
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
              targets: list[str] = None,
              **kwargs):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}', training_data).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        self.preds = self.model.predict(data).values
        return self.preds

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
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


class RegularizedRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data).fit_regularized(**kwargs)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        self.preds = self.model.predict(data).values
        return self.preds

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)
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
              targets: list[str] = None,
              **kwargs):
        X = training_data[features]
        self.model = pygam.GAM(verbose=True, **kwargs)
        print(f'{bcolors.OKGREEN}GAM grid search..')
        self.model.gridsearch(training_data[features].values, training_data[targets].values, keep_best=True,
                              progress=True)
        self.model.fit(training_data[features].values, training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        self.preds = self.model.predict(data[features])
        return self.preds

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)
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
              targets: list[str] = None,
              **kwargs):
        self.model = sm.RLM.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data,
                                         M=sm.robust.norms.HuberT(), **kwargs).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        self.preds = self.model.predict(data).values
        return self.preds

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)
        preds = self.predict(testing_data)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.2f}') for k, v in results.items()]
        return results


class PolynomialRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        poly = np.polyfit()


class NNRegressor(RegressorModel):

    @staticmethod
    def denormalized_mae(y_true, y_preds):
        return tf.abs(y_true - y_preds) * 251.01737952758302 + 92.8677986740517

    def build(self, input_shape, dense_width, dense_depth, dense_shrink, dropout):
        inp = Input(shape=(input_shape))
        x = Dense(dense_width, activation='swish')(inp)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        for _ in range(dense_depth):
            x = Dense(round(dense_width * dense_shrink), activation='swish')(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        out = Dense(1, activation='linear')(x)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer='Adam',
                           loss=tf.keras.losses.Huber(),
                           metrics=tf.keras.metrics.RootMeanSquaredError())

        print(self.model.summary())

    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        # convert dfs to ndarrays
        x_train = training_data[features].iloc[int(len(training_data) * .2):].values.astype('float32')
        y_train = training_data[targets].iloc[int(len(training_data) * .2):].values.astype('float32')

        x_eval = training_data[features].iloc[:int(len(training_data) * .2)].values.astype('float32')
        y_eval = training_data[targets].iloc[:int(len(training_data) * .2)].values.astype('float32')

        print(f'train samples: {x_train.shape[0]}\ntest samples: {x_eval.shape[0]}')

        # modeling
        self.build(x_train.shape[-1], **kwargs)

        os.makedirs('./tmp', exist_ok=True)

        checkpoint = ModelCheckpoint(f'./tmp/NN_weights_best.h5',
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', patience=150)

        hist = self.model.fit(x=x_train,
                              y=y_train,
                              validation_data=(x_eval, y_eval),
                              epochs=1000,
                              batch_size=1000,
                              callbacks=[checkpoint, es])

        self.model.load_weights('./tmp/NN_weights_best.h5')

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('loss vs epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        return hist,

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       denorm: tuple = None,
                       verbose: bool = True,
                       **kwargs):
        self.denorm = denorm

        self.train(training_data, features, targets, **kwargs)
        preds = self.predict(testing_data, features)

        preds_denorm = denormalize(preds, *denorm)

        y_trues = testing_data[targets].values.astype('float32')
        y_trues_denorm = denormalize(y_trues, *denorm)

        results = score_regression(y_trues_denorm, preds_denorm)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]
        return results


# ********************** CLS ************************


class NaiveClassifier(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        pass

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return np.random.randint(low=0, high=self.num_classes, size=(len(data),))

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):

        preds = self.predict(testing_data, features)

        results = score_classification(testing_data[targets].values, preds)

        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class DecisionTree(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = DecisionTreeClassifier(random_state=123, **kwargs).fit(training_data[features].values,
                                                                            training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):

        self.train(training_data, features, targets, **kwargs)
        preds = self.predict(testing_data, features)
        results = score_classification(testing_data[targets].values, preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class RandomForest(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = RandomForestClassifier(random_state=123, **kwargs)
        self.model.fit(training_data[features].values, training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)

        preds = self.predict(testing_data, features)

        results = score_classification(testing_data[targets].values, preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class LSVC(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = LinearSVC(random_state=123, **kwargs)
        self.model.fit(training_data[features].values, training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)

        preds = self.predict(testing_data, features)

        results = score_classification(testing_data[targets].values, preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class SVClassifier(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = SVC(random_state=123, **kwargs)
        self.model.fit(training_data[features].values, training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)

        preds = self.predict(testing_data, features)

        results = score_classification(testing_data[targets].values, preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class Boosting(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)
        self.model.fit(training_data[features].values, training_data[targets].values)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        return self.model.predict(data[features].values)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):
        self.train(training_data, features, targets, **kwargs)

        preds = self.predict(testing_data, features)

        results = score_classification(testing_data[targets].values, preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(testing_data[targets].values, preds)

        return results


class NNClassifier(ClassifierModel):
    def build(self, input_shape, dense_width, dense_depth, dense_shrink, dropout):
        inp = Input(shape=(input_shape))
        x = Dense(dense_width, activation='swish')(inp)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        for _ in range(dense_depth):
            x = Dense(round(dense_width * dense_shrink), activation='swish')(x)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
        out = Dense(self.num_classes, activation='linear')(x)

        if self.num_classes == 2:
            loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, label_smoothing=.15)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=loss,
                           metrics='Accuracy')

        print(self.model.summary())

    def train(self, training_data: pd.DataFrame = None,
              testing_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None,
              **kwargs):

        x_train = training_data[features].values.astype('float32')
        y_train = training_data[targets].values.astype('float32')
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes, dtype='float32')

        x_eval = testing_data[features].values.astype('float32')
        y_eval = testing_data[targets].values.astype('float32')
        y_eval = tf.keras.utils.to_categorical(y_eval, num_classes=self.num_classes, dtype='float32')

        print(f'train samples: {x_train.shape[0]}\ntest samples: {x_eval.shape[0]}')

        self.build(input_shape=x_train.shape[-1], **kwargs)

        os.makedirs('./tmp', exist_ok=True)

        checkpoint = ModelCheckpoint(f'./tmp/NN_class_weights_best.h5',
                                     monitor='val_Accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        es = EarlyStopping(monitor='val_Accuracy', patience=200, mode='max')

        hist = self.model.fit(x=x_train,
                              y=y_train,
                              validation_data=(x_eval, y_eval),
                              epochs=1000,
                              batch_size=1600,
                              callbacks=[checkpoint, es],
                              # class_weight=class_weights
                              )

        self.model.load_weights('./tmp/NN_class_weights_best.h5')

        self.model.evaluate(x=x_eval, y=y_eval)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.show()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None,
                **kwargs) -> np.ndarray:
        preds = self.model.predict(data[features].values)
        np.save('preds.npy', preds)
        return np.argmax(preds, axis=1)

    def train_and_test(self, training_data: pd.DataFrame = None,
                       testing_data: pd.DataFrame = None,
                       features: list[str] = None,
                       targets: list[str] = None,
                       verbose: bool = True,
                       **kwargs):

        self.train(training_data, testing_data, features, targets, **kwargs)
        y_preds = self.predict(testing_data, features)

        y_trues = testing_data[targets].values.astype('float32')
        np.save('trues.npy', y_trues)

        results = score_classification(y_trues, y_preds)
        if verbose:
            print(f'\n{self.__class__.__name__}:')
            [print(f'{k}: {v:.6f}') for k, v in results.items()]

            plot_confusion_mtarix(y_trues, y_preds)

        return results
