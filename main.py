import pandas as pd

from models import *
from common import *

# TODO: -regressors: RANSAC, GAM, polyfit, DNNs, (SNNs)
# TODO: -classifers: naive, SVM, DTRee, RF, XGB, DNN
# TODO: -profiling
# todo: -auto evaluator
# todo: -more features
# todo: -hyperparameter plots -> only for best models


if __name__ == '__main__':
    features = ['average_rate_weapon', 'pistol_usage', 'difficulty']
    targets = ['playtime']

    # input_data = 'C:/KBData/04_tmp/elte/l4d2_player_stats_final_cleaned.parquet'
    # df = pd.read_parquet(input_data)
    #
    # df, *denorm = normalize(df, df.columns, targets[0])
    #
    # df.to_parquet('C:/KBData/04_tmp/elte/l4d2_player_stats_final_cleaned_normalized.parquet')
    #
    # with open('C:/KBData/04_tmp/elte/denorm.pkl', 'wb') as file:
    #     pickle.dump(denorm, file)

    df = pd.read_parquet('C:/KBData/04_tmp/elte/l4d2_player_stats_final_cleaned_normalized.parquet')

    with open('C:/KBData/04_tmp/elte/denorm.pkl', 'rb') as file:
        denorm = pickle.load(file)

    df_train, df_test = split_data(df, .75)

    all_features = [col for col in df.columns if 'usage' in col or 'rate' in col] + ['difficulty']

    print(all_features)

    model = NaiveRegressor()
    model.train_and_test(training_data=df_train,
                         features=features,
                         testing_data=df_test,
                         targets=targets,
                         denorm=denorm
    )

    model = LinearRegressor()
    model.train_and_test(training_data=df_train,
                         features=features,
                         testing_data=df_test,
                         targets=targets,
                         denorm=denorm
    )

    model = GAM()
    model.train_and_test(training_data=df_train,
                         features=features,
                         testing_data=df_test,
                         targets=targets,
                         denorm=denorm
    )

    model = NNRegressor()
    model.train_and_test(training_data=df_train,
                         features=all_features,
                         testing_data=df_test,
                         targets=targets,
                         denorm=denorm)

    print("\nDONE")
