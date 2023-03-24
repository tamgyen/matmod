import abc
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from common import bcolors, EPSILON


def score_regression(y_true: np.ndarray = None, y_preds: np.ndarray = None):
    assert y_true.size == y_preds.size, f'{bcolors.FAIL}Size mismatch in y_true and y_pred\n'
    y_preds = np.squeeze(y_preds)
    y_true = np.squeeze(y_true)

    return {metric.__name__: round(metric().score(y_true, y_preds), 8) for metric in RegressionMetric.__subclasses__()}


def score_classification(y_true: np.ndarray = None, y_preds: np.ndarray = None):
    assert y_true.size == y_preds.size, f'{bcolors.FAIL}Size mismatch in y_true and y_pred\n'
    y_preds = np.squeeze(y_preds)
    y_true = np.squeeze(y_true)

    report = classification_report(y_true, y_preds, output_dict=True)

    return {'Accuracy': round(report['accuracy'], 8), 'F1Score': round(report['weighted avg']['f1-score'], 8)}


def plot_confusion_mtarix(y_true: np.ndarray = None, y_preds: np.ndarray = None):
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


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


class RootMeanSquaredError(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


class MeanAbsoluteError(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.abs(y_true - y_pred))


class R2(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        a = sum(np.square(y_true - y_pred))
        b = sum(np.square(y_true - np.mean(y_true)))
        return 1 - (a / b)


class FitPercent(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        # also called Normalized Root Mean Squared Error (NRMSE): referring to
        # https://www.mathworks.com/help/ident/ug/model-quality-metrics.html

        return 100 * (1 - np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true - np.mean(y_true)))


class CosineSimilarity(RegressionMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))


# ********************** CLS ************************


class KLDivergence(ClassificationMetric):
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_true * np.log(y_true / y_pred)

# class Accuracy(ClassificationMetric):
#     def score(self, y_true: np.ndarray, y_pred: np.ndarray):
#         scores = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
#         return scores['accuracy']

# class F1Score(Metric):
#     pass
