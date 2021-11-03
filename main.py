from data_preprocess import data_preprocess
from utils import load_csv, plot_feature_importance
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # read the data
    train, train_labels, test_og = load_csv('train_features.csv', 'train_salaries.csv','test_features.csv')
    train = train[['jobType', 'yearsExperience','milesFromMetropolis','industry','degree','major']]
    test = test_og[['jobType', 'yearsExperience','milesFromMetropolis','industry','degree','major']]

    # clean the data
    cleaned_train = data_preprocess.impute_process(train)
    cleaned_test = data_preprocess.impute_process(test)

    # feature engineer
    train_final, test_final = data_preprocess.encoding(cleaned_train,cleaned_test)

    # split the training into train and validation
    x, x_test, v, v_test = train_test_split(train_final, train_labels, test_size=0.2, random_state=42)

    # set parameters
    parameters = {
    'n_estimators': 2500,
    'num_leaves': 16,
    'learning_rate': 0.05,
    'colsample_bytree': 0.5,
    'max_depth': None,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'min_split_gain': 0.0,
    'min_child_weight': 5,
    'boost_from_average': True,
    'early_stopping_rounds': 200,
    'huber_delta': 1.0,
    'min_child_samples': 10,
    'objective': 'regression_l2',
    'subsample_for_bin': 50000,
    "metric": 'rmse'
    }


    ##################################################################################
    #                                                                                #
    #                      LGBMRegressor build-on function                           #
    #                                                                                #
    ##################################################################################
    lgbm = LGBMRegressor(**parameters)
    lgbm.fit(x, v, eval_set=[(x_test, v_test)])

    # predict
    y_pred = lgbm.predict(test_final, num_iteration=lgbm.best_iteration_)

    # save as csv file
    res = pd.DataFrame({'jobId': test_og.jobId, 'salary': y_pred})
    res.to_csv('test_salaries.csv', index=False)

    # feature importances 
    plot_feature_importance(lgbm.feature_importances_, x_test.columns, 'lightGBM')

if __name__ == '__main__':
    main()