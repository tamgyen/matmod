from metrics import *
from models import *


# TODO: -regressors: GAM, NNs, SNNs
# TODO: -classifers: naive, SVM, DTRee, RF, XGB
# TODO: -profiling
# todo: -auto evaluator
# todo: EDA
# todo:


def score_regression(y_true: np.ndarray = None, y_preds: np.ndarray = None):
    return {metric.__name__: metric().score(y_true, y_preds) for metric in RegressionMetric.__subclasses__()}


if __name__ == '__main__':
    input_data = 'C:/Dev/ELTE_AI/mat_mod/data/l4d2_player_stats_final.parquet'

    df = pd.read_parquet(input_data)
    df = df[df['Playtime'] != 0]

    model = NaiveRegressor()
    model.train(training_data=df, features=['Pistol_Kills', 'Pistol_Shots'], targets=['Playtime'])
    preds_naive = model.predict(data=df, features=['Kills', 'Shots'])

    y_trues = df['Playtime'].values
    results_naive = score_regression(y_trues, preds_naive)
    print('\nNAIVE:')
    [print(f'{k}: {v:.2f}') for k, v in results_naive.items()]

    model_reg = LinearRegressor()
    model_reg.train(training_data=df, features=['Pistol_Kills', 'Pistol_Shots'], targets=['Playtime'])
    preds_linear = model_reg.predict(data=df)

    results_linear = score_regression(y_trues, preds_linear)

    print('\nLINEAR:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]

    model_reg = LassoRegressor()
    model_reg.train(training_data=df, features=['Pistol_Kills', 'Pistol_Shots'], targets=['Playtime'])
    preds_linear = model_reg.predict(data=df)

    results_linear = score_regression(y_trues, preds_linear)

    print('\nLASSO:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]

    model_reg = RobustRegressor()
    model_reg.train(training_data=df, features=['Pistol_Kills', 'Pistol_Shots'], targets=['Playtime'])
    preds_linear = model_reg.predict(data=df)

    results_linear = score_regression(y_trues, preds_linear)

    print('\nGAM:')
    [print(f'{k}: {v:.2f}') for k, v in results_linear.items()]


