from utils.bayesian_encoding import Bayesian_Encoding
from utils import utils
from utils import layer2
from utils.xgb_fn import *

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
%matplotlib inline

from jupyterthemes import jtplot
jtplot.reset()

train, test = utils.load_data()
X_cat = pd.read_csv('data/cat_train.csv')
X_cat_test = pd.read_csv('data/cat_test.csv')
X_percent = pd.read_csv('data/percent_train.csv')
X_percent_test = pd.read_csv('data/percent_test.csv')
X_pca = pd.read_csv('data/pca100_train.csv')
X_pca_test = pd.read_csv('data/pca100_test.csv')


X = train.drop(['id', 'target'], axis=1)
X_test = test.drop('id', axis=1)

X['ps_reg_03'] = (X.ps_reg_03**2)*16*100
X['ps_car_15'] = X.ps_car_15**2
X['ps_car_14'] = (X.ps_car_14**2)*10000
X['ps_car_13'] = X.ps_car_13**2*100000
X['ps_car_13'] = X.ps_car_13.round(1)
X['ps_car_12'] = X.ps_car_12**2*100
X['ps_car_12'] = X.ps_car_12.round(0)*100

X_test['ps_reg_03'] = (X_test.ps_reg_03**2)*16*100
X_test['ps_car_15'] = X_test.ps_car_15**2
X_test['ps_car_14'] = (X_test.ps_car_14**2)*10000
X_test['ps_car_13'] = X_test.ps_car_13**2*100000
X_test['ps_car_13'] = X_test.ps_car_13.round(1)
X_test['ps_car_12'] = X_test.ps_car_12**2*100
X_test['ps_car_12'] = X_test.ps_car_12.round(0)*100

X = X.drop(X.columns[X.columns.str.startswith('ps_calc')], axis=1)
X = X.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin',
                'ps_ind_18_bin','ps_car_03_cat','ps_car_10_cat'], axis=1)
X = pd.get_dummies(X, columns=X.columns[X.columns.str.endswith('cat')==True], dummy_na=True)

X_test = X_test.drop(X_test.columns[X_test.columns.str.startswith('ps_calc')], axis=1)
X_test = X_test.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin',
                'ps_ind_18_bin','ps_car_03_cat','ps_car_10_cat'], axis=1)
X_test = pd.get_dummies(X_test, columns=X_test.columns[X_test.columns.str.endswith('cat')==True], dummy_na=True)

X_cat = X_cat[['ps_reg_03_cat', 'ps_car_13_cat', 'ps_reg_01_cat', 'ps_reg_02_cat']]
X_cat_test = X_cat_test[['ps_reg_03_cat', 'ps_car_13_cat', 'ps_reg_01_cat', 'ps_reg_02_cat']]

X_percent = X_percent.drop(['id', 'target'], axis=1)
X_percent_test = X_percent_test.drop(['id', 'target'], axis=1)
X_percent.columns = ['percent_'+i for i in X_percent.columns]
X_percent_test.columns = ['percent_'+i for i in X_percent_test.columns]

X_pca = X_pca.drop(['id', 'target'], axis=1)
X_pca_test = X_pca_test.drop(['id', 'target'], axis=1)