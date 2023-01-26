import numpy as np
import pandas as pd
import pickle


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
    df_samp = df.sample(frac=1, random_state=123).reset_index(drop=True)
    df_train = df_samp[:split_id]
    df_test = df_samp[split_id:].reset_index(drop=True)

    return df_train, df_test


def denormalize(data: np.ndarray, mean, std):
    return data*std + mean


def normalize(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    target_mean = df[target_column].mean(axis=0)
    target_std = df[target_column].std(axis=0)

    df[feature_columns] = df[feature_columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    return df, target_mean, target_std
