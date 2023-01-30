import itertools
import numpy as np
import pandas as pd
import pickle
import tqdm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


EPSILON = 1e-7


def split_data(df: pd.DataFrame, split: float):
    split_id = round(split * len(df))
    df_samp = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_samp[:split_id]
    df_test = df_samp[split_id:].reset_index(drop=True)

    return df_train, df_test


def denormalize(data: np.ndarray, mean, std):
    return data * std + mean


def normalize(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    target_mean = df[target_column].mean(axis=0)
    target_std = df[target_column].std(axis=0)

    df[feature_columns + [target_column]] = df[feature_columns + [target_column]].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    return df, target_mean, target_std


def haty_norm2(X, y):
    q, r = np.linalg.qr(X)
    return (np.dot(q.T, y) ** 2).sum()


def best_k(X, y, k):
    best_v = 0
    best_c = None
    for c in itertools.combinations(range(X.shape[1]), k):
        v = haty_norm2(X[:, c], y)
        if v > best_v:
            best_v = v
            best_c = c
    return best_c, best_v


def best_subset(X, y, max_k=8, min_k=1):
    col_names = X.columns
    y, X = np.asarray(y), np.asarray(X)
    y = y - y.mean(axis=0)
    X = X - X.mean(axis=0)
    q, r = np.linalg.qr(X)
    TSS = (y ** 2).sum()
    y = np.dot(q.T, y)

    def result(x):
        c, v = x
        return {
            'num_pred': len(c),
            'predictors': [col_names[i] for i in c],
            'R2': v / TSS
        }

    return pd.DataFrame([result(best_k(r, y, k))
                         for k in tqdm.tqdm(range(min_k, min(len(col_names), max_k) + 1))]).set_index('num_pred')


def forward_stepwise_selection(y, X):
    col_names = list(X.columns)
    y, X = np.asarray(y), np.asarray(X)

    y = y - y.mean(axis=0)
    X = X - X.mean(axis=0)

    TSS = (y ** 2).sum()

    q, X = np.linalg.qr(X)
    y = np.dot(q.T, y)

    tv = 0
    result = []
    predictors = []
    while len(col_names) > 0:
        (c,), v = best_k(y=y, X=X, k=1)
        tv += v
        predictors.append(col_names.pop(c))
        result.append(dict(num_pred=len(predictors), predictors=predictors.copy(), r2=tv / TSS))

        x, X = X[:, c], X[:, np.arange(X.shape[1]) != c]

        x = x / np.linalg.norm(x)
        y = y - np.dot(y, x) * x
        X = X - np.outer(x, np.dot(x.T, X))
    return pd.DataFrame(result).set_index('num_pred')


def backward_stepwise_selection(y, X):
    col_names = list(X.columns)
    y, X = np.asarray(y), np.asarray(X)

    y = y - y.mean(axis=0)
    X = X - X.mean(axis=0)

    TSS = (y ** 2).sum()

    q, X = np.linalg.qr(X)
    y = np.dot(q.T, y)

    tv = 0
    result = []
    while len(col_names) > 1:
        c, v = best_k(y=y, X=X, k=len(col_names) - 1)

        col_names = [col_names[i] for i in c]
        result.append(dict(num_pred=len(c),
                           predictors=col_names,
                           r2=v / TSS))

        X = X[:, c]
    return pd.DataFrame(result).set_index('num_pred')
