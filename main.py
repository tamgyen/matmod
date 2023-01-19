from metrics import score_regression
from models import *


# TODO: -regressors: RAMSAC, GAM, polyfit, DNNs, (SNNs)
# TODO: -classifers: naive, SVM, DTRee, RF, XGB, DNN
# TODO: -profiling
# todo: -auto evaluator

if __name__ == '__main__':
    input_data = 'C:/Dev/ELTE_AI/mat_mod/data/l4d2_player_stats_final_cleaned.parquet'
    df = pd.read_parquet(input_data)

    features = ['ranged_kps', 'knife_kills']
    targets = ['playtime']

    y_trues = df['playtime'].values

    model = NaiveRegressor()
    model.train(df, features, targets)
    preds_naive = model.predict(df)

    results_naive = score_regression(y_trues, preds_naive)
    print('\nNAIVE:')
    [print(f'{k}: {v:.2f}') for k, v in results_naive.items()]

    model_reg = LinearRegressor()
    model_reg.train(df, features, targets)
    preds_linear = model_reg.predict(df)

    results_linear = score_regression(y_trues, preds_linear)
    print('\nLINEAR:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]

    model_reg = LassoRegressor()
    model_reg.train(df, features, targets)
    preds_linear = model_reg.predict(df)

    results_linear = score_regression(y_trues, preds_linear)
    print('\nLASSO:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]

    model_reg = RobustRegressor()
    model_reg.train(df, features, targets)
    preds_linear = model_reg.predict(df)

    results_linear = score_regression(y_trues, preds_linear)
    print('\nRobust:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]

    # model_nn = NNRegressor()
    # model_nn.train(training_data=df, features=['ranged_kps', 'knife_usage'], targets=['playtime'])

    print("DONE")
