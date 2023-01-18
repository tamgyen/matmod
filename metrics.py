import abc
import numpy as np

from common import bcolors, EPSILON


class Metric(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'score') and
                callable(subclass.score) or NotImplemented)

    @abc.abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        assert y_true.shape == y_pred.shape, f'{bcolors.FAIL}shapes {y_true} and {y_pred} dont match!'
        raise NotImplementedError


class RegressionMetric(Metric, abc.ABC):
    pass


class ClassificationMetric(Metric, abc.ABC):
    pass


class MeanSquaredError(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.power(y_true - y_pred, 2))


class MeanAbsoluteError(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.abs(y_true - y_pred))


class MeanAbsolutePercentageError(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true[y_true == 0] = EPSILON
        return np.mean(np.abs((y_true - y_pred)) / y_true) * 100


class FitPercent(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        # also called Normalized Root Mean Squared Error (NRMSE): referring to
        # https://www.mathworks.com/help/ident/ug/model-quality-metrics.html

        return 100 * (1 - (np.linalg.norm(y_true - y_pred)) / np.linalg.norm(y_true - np.mean(y_true)))


class CosineSimilarity(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))


# ********************** CLS ************************


class KLDivergence(ClassificationMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_true * np.log(y_true / y_pred)


class Accuracy(ClassificationMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        differing_labels = np.count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
        print(score)


class F1Score(Metric):
    pass
