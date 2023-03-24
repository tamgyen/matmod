import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import split_data, normalize
from metrics import score_classification, plot_confusion_mtarix
from models import NNClassifier

import xgboost as xgb

from models import *


df = pd.read_parquet('./data/l4d2_player_stats_final_cleaned.parquet')
features = [col for col in df.columns if 'usage' in col or 'rate' in col]

df, *denorm = normalize(df, features, 'playtime')

df['difficulty_binary'] = np.where(df['difficulty'].values <=1 , 0, 1)

targets = ['difficulty']
num_classes = 4

df_train, df_test = split_data(df, .75)

model = NaiveClassifier(num_classes)
results = model.train_and_test(df_train, df_test, features, targets)
print(f'NAIVE SCORE: {results}')

model = DecisionTree(num_classes)
results = model.train_and_test(df_train, df_test, features, targets)

print(f'SINGLE TREE SCORE: {results}')

model = RandomForest(num_classes)
results = model.train_and_test(df_train, df_test, features, targets, max_features=10)

print(f'FOREST SCORE: {results}')

model = SVClassifier(num_classes)
results = model.train_and_test(df_train, df_test, features, targets)
#
# print(f'SVC SCORE: {results}')
#
# model = LSVC(num_classes)
# results = model.train_and_test(df_train, df_test, features, targets)
#
# print(f'LSVC SCORE: {results}')

# model = Boosting(num_classes)
# results = model.train_and_test(df_train, df_test, features, targets)
#
# print(f'Boosting SCORE: {results}')





# # *********FOREST***************
# m = round(math.sqrt(len(features)-1))
# random_forest = RandomForestClassifier(max_features=m, random_state=123)
# random_forest.fit(X_train, y_train)
#
# y_pred = random_forest.predict(X_test)
# score = score_classification(y_true, y_pred)
# print(f'RANDOM FOREST SCORE: {score}')
#
# plot_confusion_mtarix(y_true, y_pred)
#
# # **********XGB********************
#
# data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
#
# xgc = xgb.XGBClassifier()
#
# xgc.fit(X_train, y_train)
#
# y_pred = xgc.predict(X_test)
# score = score_classification(y_true, y_pred)
# print(f'XGBOOST SCORE: {score}')
#
# plot_confusion_mtarix(y_true, y_pred)


#
model = NNClassifier(num_classes=4)
model.train_and_test(training_data=df_train,
                     features=features,
                     testing_data=df_test,
                     targets=targets,
                     denorm=denorm,
                     dense_width=80,
                     dense_depth=3,
                     dense_shrink=.5,
                     dropout=.4)