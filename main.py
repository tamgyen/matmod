from models import *
from common import *

# TODO: -regressors: RAMSAC, GAM, polyfit, DNNs, (SNNs)
# TODO: -classifers: naive, SVM, DTRee, RF, XGB, DNN
# TODO: -profiling
# todo: -auto evaluator
# todo: -more features
# todo: -hyperparameter plots -> only for best models


if __name__ == '__main__':
    input_data = 'C:/Dev/ELTE_AI/mat_mod/data/l4d2_player_stats_final_cleaned.parquet'
    df = pd.read_parquet(input_data)

    features = ['average_rate_weapon', 'difficulty']
    targets = ['playtime']

    df, *denorm = normalize(df, df.columns, targets[0])

    df_train, df_test = split_data(df, .75)

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

    # model_reg = LinearRegressor()
    # model_reg.train(df, features, targets)
    # preds_linear = model_reg.predict(df)
    #
    # results_linear = score_regression(y_trues, preds_linear)
    # print('\nLINEAR:')
    # [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]
    #
    # model_reg = LassoRegressor()
    # model_reg.train(df, features, targets)
    # preds_linear = model_reg.predict(df)
    #
    # results_linear = score_regression(y_trues, preds_linear)
    # print('\nLASSO:')
    # [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]
    #
    # model_reg = RobustRegressor()
    # model_reg.train(df, features, targets)
    # preds_linear = model_reg.predict(df)
    #
    # results_linear = score_regression(y_trues, preds_linear)
    # print('\nRobust:')
    # [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]
    #
    # model_nn = NNRegressor()
    # model_nn.train(training_data=df, features=features, targets=targets)
    #
    # print("\nDONE")
