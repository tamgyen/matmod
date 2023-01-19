import abc
import numpy as np
import pandas as pd
from time import perf_counter
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

import tensorflow as tf
from keras.layers import Dense, Input


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


class LinearRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}', training_data).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values


class LassoRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        self.model = sm.OLS.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data).fit_regularized()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values


class GAM(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        # TODO:
        x_spline = training_data[features]
        bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])
        self.model = GLMGam.from_formula(f'{targets[0]} ~ {" + ".join(features)}',
                                         training_data, smoother=bs).fit()

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        return self.model.predict(data).values


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


class NNRegressor(RegressorModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        # normalize
        df_norm = training_data[features + targets].copy()
        # df_norm = df_norm.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # split
        split = .8
        split_id = round(split * len(training_data))
        df_norm = df_norm.sample(frac=1, random_state=333).reset_index(drop=True)
        df_train = df_norm[:split_id]
        df_test = df_norm[split_id:].reset_index(drop=True)

        # convert dfs to ndarrays
        x_train = df_train[features].values.astype('float32')
        y_train = df_train[targets].values.astype('float32')

        x_eval = df_test[features].values.astype('float32')
        y_eval = df_test[targets].values.astype('float32')

        # modeling
        inp = Input(shape=(x_train.shape[-1]))
        x = Dense(4, activation='relu')(inp)
        x = Dense(2, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.Huber(),
                           metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
                                    tf.keras.metrics.MeanAbsoluteError()])

        # fit
        hist = self.model.fit(x=x_train,
                              y=y_train,
                              validation_data=(x_eval, y_eval),
                              epochs=100,
                              batch_size=1000)

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        pass


# ********************** CLS ************************


class NaiveClassifier(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        pass

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        pass
