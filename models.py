import abc

import numpy as np
import pandas as pd
from time import perf_counter

import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines


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


# ********************** CLS ************************


class NaiveClassifier(ClassifierModel):
    def train(self, training_data: pd.DataFrame = None,
              features: list[str] = None,
              targets: list[str] = None):
        pass

    def predict(self, data: pd.DataFrame = None,
                features: list[str] = None) -> np.ndarray:
        pass

