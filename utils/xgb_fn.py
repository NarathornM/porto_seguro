from constants import *
from utils.gini import *

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

def xgb_skf_train(X, y, params, nfolds=5, model_path_name='xgb_models/xgb', random_seed=123, shuffle_split=False,
    early_stop = 100, verbose=100):
    X = X.values
    y = y.values
    score = []
    modelname = model_path_name
    model_num = 1
    skf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle_split, random_state=random_seed)

    for train_index, val_index in skf.split(X, y):
        # print('Training xgb Fold:{}/{}'.format(model_num, nfolds))
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # Set Var for XGB
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_val = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(xg_train, 'train'), (xg_val, 'test')]
        #
        xgb_skf = xgb.train(params, xg_train, 10000, watchlist, early_stopping_rounds=early_stop
                            , feval=gini_xgb, maximize=True, verbose_eval=verbose)
        print('[Fold {}/{} Gini score: {}]'.format(model_num, nfolds, xgb_skf.best_score))
        score.append(xgb_skf.best_score)
        
        joblib.dump(xgb_skf, '{}_{}.pkl'.format(model_path_name, model_num))
        model_num += 1
    return print('[Successfully trained, Average Gini score: {} \n]'.format(np.mean(score)))

def xgb_skf_predict(X_test, model_path_name='xgb_models/xgb'):
    model_list = glob.glob('{}*.pkl'.format(model_path_name))
    xg_test = xgb.DMatrix(X_test.values)
    pred = np.zeros(X_test.shape[0], )

    for model in model_list:
        xgb_pred = joblib.load(model)
        pred_lap = xgb_pred.predict(xg_test)
        pred += pred_lap
    
    return pred