import pandas as pd

import metrics
from models import *
from common import *

# TODO: -regressors: RANSAC, GAM, polyfit, DNNs, (SNNs)
# TODO: -classifers: naive, SVM, DTRee, RF, XGB, DNN
# TODO: -profiling
# todo: -auto evaluator
# todo: -more features
# todo: -hyperparameter plots -> only for best models


if __name__ == '__main__':
    # features = ['m60_rate',
    #             'katana_rate',
    #             'machete_rate',
    #             'pistol_usage',
    #             'chainsaw_usage',
    #             'katana_usage',
    #             'molotov_rate',
    #             'pipe_bomb_rate',
    #             'bile_jar_rate',
    #             'average_rate_utility']

    features = ['assault_rifle_usage',
                'average_rate_weapon',
                'chainsaw_rate',
                'chainsaw_usage',
                'combat_shotgun_usage',
                'desert_rifle_usage',
                'difficulty',
                'fire_axe_rate',
                'fire_axe_usage',
                'hunting_rifle_usage',
                'katana_rate',
                'katana_usage',
                'machete_rate',
                'magnum_rate',
                'magnum_usage',
                'max_rate_utility',
                'pistol_rate',
                'pistol_usage',
                'pump_shotgun_usage',
                'tactical_shotgun_usage']

    targets = ['playtime']

    df = pd.read_parquet('./data/l4d2_player_stats_final_cleaned_normalized.parquet')

    with open('./data/denorm.pkl', 'rb') as file:
        denorm = pickle.load(file)

    df_train, df_test = split_data(df, .75)


    # model = NaiveRegressor()
    # resut = model.train_and_test(training_data=df_train,
    #                              features=features,
    #                              testing_data=df_test,
    #                              targets=targets,
    #                              denorm=denorm
    #                              )
    #
    # model = LinearRegressor()
    # model.train_and_test(training_data=df_train,
    #                      features=features,
    #                      testing_data=df_test,
    #                      targets=targets,
    #                      denorm=denorm
    #                      )
    #
    # model = RegularizedRegressor()
    # model.train_and_test(training_data=df_train,
    #                      features=features,
    #                      testing_data=df_test,
    #                      targets=targets,
    #                      denorm=denorm,
    #                      verbose=True,
    #                      method='elastic_net',
    #                      alpha=1,
    #                      L1_wt=.5,
    #                      )

    # model = GAM()
    # model.train_and_test(training_data=df_train,
    #                      features=features,
    #                      testing_data=df_test,
    #                      targets=targets,
    #                      denorm=denorm
    #                      )
    #
    model = NNRegressor()
    model.train_and_test(training_data=df_train,
                         features=features,
                         testing_data=df_test,
                         targets=targets,
                         denorm=denorm,
                         dense_width=80,
                         dense_depth=3,
                         dense_shrink=.5,
                         dropout=.4)

    # print("\nDONE")

    # model = NaiveClassifier(num_classes=4)
    # scores = model.train_and_test(training_data=None, testing_data=df_test, targets=targets, features=features)

    # model = NNClassifier(num_classes=4)
    # model.train_and_test(training_data=df_train,
    #                      features=features,
    #                      testing_data=df_test,
    #                      targets=targets,
    #                      denorm=denorm)
