import os
from constants import *
from utils.utils import *
from utils.gini import *
from utils.xgb_fn import*
from utils import utils

import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, KFold


folds = 5

class Layer1Train():
    def __init__(self, model, params, MODEL_NAME, cv='skf', fillna='median', my_features=False, smoothing=False, drop_stupid=False, cat_transform=False, data_transform=False, recon_category=False,
    feature_interactions=False, engineer_stats=False, kinetic_feature=False, seed=88, shuffle=False):
        self.model = model
        self.params = params
        self.MODEL_NAME = MODEL_NAME
        self.drop_stupid = drop_stupid
        self.cat_transform = cat_transform
        self.data_transform = data_transform
        self.recon_category = recon_category
        self.feature_interactions = feature_interactions
        self.engineer_stats = engineer_stats
        self.kinetic_feature = kinetic_feature
        self.fillna = fillna
        self.smoothing = smoothing
        self.cv = cv
        self.my_features = my_features
        self.seed = seed
        self.shuffle = shuffle

    def make_oof(self, model, params, X_train, y_train, X_test, MODEL_NAME):
        kfolds = folds
        avg_score = 0
        oof_train = np.zeros(X_train.shape[0], )
        oof_test = np.zeros(X_test.shape[0], )
        oof_test_skf = np.empty((kfolds, X_test.shape[0]))

        if self.cv == 'skf':
            cross_val = StratifiedKFold(n_splits=kfolds, shuffle=self.shuffle, random_state=self.seed)
            print('Training using StratifiedKFold')
        elif self.cv == 'kf':
            cross_val = KFold(n_splits=kfolds, shuffle=self.shuffle, random_state=self.seed)
            print('Training using KFold')
        else:
            print("Please specify type of cv ('kf', 'skf')")

        
        for i, (train_index, val_index) in enumerate(cross_val.split(X_train, y_train)):

            print('[Fold {}/{} START]'.format(i + 1, kfolds))

            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index,:]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            if self.smoothing:
                X_tr, X_val, X_test = fold_smoothing_encode(X_tr, X_val, X_test, y_tr)

            if 'xgb' in MODEL_NAME:
                print('Xgboost is training')
                # Set Var for XGB
                xg_train = xgb.DMatrix(X_tr, label=y_tr)
                xg_val = xgb.DMatrix(X_val, label=y_val)
                xg_test = xgb.DMatrix(X_test)
                watchlist = [(xg_train, 'train'), (xg_val, 'test')]
                # train xgb
                xgb_skf = xgb.train(params, xg_train, 10000, watchlist, early_stopping_rounds=100
                                    , feval=gini_xgb, maximize=True, verbose_eval=100)
                avg_score += xgb_skf.best_score / kfolds
                oof_train[val_index] = xgb_skf.predict(xg_val)
                oof_test += xgb_skf.predict(xg_test) / kfolds

                print('[Fold {}/{} Gini score: {}]'.format(i+1, kfolds, xgb_skf.best_score))
                print('[Fold {}/{} END]'.format(i+1, kfolds))

            elif 'lgb' in MODEL_NAME:
                print('Lightgbm is training')
                # Set Var for lgb
                lgb_train = lgb.Dataset(X_tr, y_tr)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
                # Train lgb
                lgb_skf = lgb.train(params, lgb_train, 10000, [lgb_train, lgb_val], verbose_eval=100, 
                                    feval=gini_lgb, early_stopping_rounds=100)

                avg_score += lgb_skf.best_score['valid_1']['gini'] / kfolds

                oof_train[val_index] = lgb_skf.predict(X_val)
                oof_test += lgb_skf.predict(X_test)/ kfolds

                print('[Fold {}/{} Gini score: {}]'.format(i+1, kfolds, lgb_skf.best_score['valid_1']['gini']))
                print('[Fold {}/{} END]'.format(i+1, kfolds))
            else:
                print('{} is training'.format(MODEL_NAME))
                model.fit(X_tr, y_tr)

                pred_val = model.predict_proba(X_val)[:,1]
                oof_train[val_index] = pred_val

                # Store 5 x numbers of observations
                oof_test += model.predict_proba(X_test)[:,1] / kfolds
                # Store average score for evaluate model
                avg_score += gini_normalized(y_val, pred_val) / kfolds

                print('[Fold {}/{} Gini Train score: {}]'.format(i+1, kfolds, gini_normalized(y_tr, model.predict_proba(X_tr)[:,1])))
                print('[Fold {}/{} Gini Valid score: {}]'.format(i+1, kfolds, gini_normalized(y_val, pred_val)))
                print('[Fold {}/{} END]'.format(i+1, kfolds))

        print('Average score: {}'.format(avg_score))


        # Export oof_train
        file_path = os.path.join(OOF_PATH, MODEL_NAME + '_train.csv')
        pd.DataFrame({MODEL_NAME: oof_train}).to_csv(file_path, index=False)

        # Export oof_test
        file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
        pd.DataFrame({MODEL_NAME: oof_test}).to_csv(file_path, index=False)

        print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))


    def train(self):
        combined = utils.load_data()
        if self.drop_stupid:
            combined = utils.drop_stupid(combined)
        # if self.bojan_features:
        #     combined = utils.bojan_engineer(combined)
        if self.engineer_stats:
            combined = utils.engineer_stats(combined)
        if self.recon_category:
            combined = utils.recon_category(combined)
        if self.cat_transform:
            combined = utils.cat_transform(combined, self.cat_transform)
        if self.data_transform:
            combined = utils.data_transform(combined, self.data_transform)
        if self.feature_interactions:
            combined = utils.feature_interactions(combined)
        if self.kinetic_feature:
            combined = pd.concat([combined, pd.read_csv('data/kinetic_combined.csv', index_col='id')], axis=1)
        if self.my_features:
            calc = utils.load_data()
            calc = calc[calc.columns[calc.columns.str.contains('calc')]]
            calc = pd.get_dummies(calc, columns=calc.columns)
            calc = calc[['ps_calc_02_0.0', 'ps_calc_02_0.1', 'ps_calc_05_3', 'ps_calc_06_7', 'ps_calc_06_10', 
            'ps_calc_07_5', 'ps_calc_08_8', 'ps_calc_08_10', 'ps_calc_10_8', 'ps_calc_11_5', 'ps_calc_11_7', 'ps_calc_11_8']]
            combined = pd.concat([combined, calc], axis=1)            

        train, test = utils.recover_train_test_na(combined, fillna=self.fillna)

        X_train = train.drop('target', axis=1)
        y_train = train.target
        X_test = test

        # Start trainning
        print('Ready to train with:')
        print('Model name ', self.MODEL_NAME)
        print('Model parameters ', self.model)
        print('X_train shape is', X_train.shape)
        print('y_train shape is', y_train.shape)
        print('X_test shape is', X_test.shape)

        self.make_oof(self.model, self.params, X_train, y_train, X_test, self.MODEL_NAME)
