from constants import *
from utils.gini import *

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# Use this to load data and do some cleaning
def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    combined = pd.concat([train.drop(['target', 'id'], axis=1), test.drop('id', axis=1)], axis=0)

    return combined

def drop_stupid(combined):
    '''
    kak >>> ps_car_11_cat ps_car_10_cat
    ps_car 1 -9 ?
    '''
    
    combined = combined.drop(combined.columns[combined.columns.str.startswith('ps_calc')], axis=1)
    # combined = combined.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin',
    #     'ps_car_11_cat', 'ps_car_10_cat'], axis=1)
    # newdrop
    combined = combined.drop(['ps_ind_10_bin','ps_car_10_cat','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin',
                        'ps_ind_18_bin','ps_car_03_cat','ps_car_10_cat'], axis=1)
    return combined

def engineer_features(combined):

        multreg = combined['ps_reg_01']*combined['ps_reg_02']*combined['ps_reg_03'] 
        ps_car_reg = combined['ps_reg_03'] * combined['ps_car_13']**2
        combined['mult'] = multreg
        combined['ps_car'] = ps_car_reg
        combined['ps_ind'] = combined['ps_ind_03'] * combined['ps_ind_15']
        combined = pd.get_dummies(combined, columns=combined.columns[combined.columns.str.endswith('cat')==True], dummy_na=True)

def kinetic(row):
        probs=np.unique(row,return_counts=True)[1]/len(row)
        kinetic=np.sum(probs**2)
        return kinetic

def kinetic_transform(combined):
    kin_ind = combined[combined.columns[combined.columns.str.contains('ind')]]
    kin_car_cat = combined[combined.columns[combined.columns.str.contains('car') & combined.columns.str.endswith('cat')]]
    kin_calc_not_bin = combined[combined.columns[combined.columns.str.contains('calc') & ~(combined.columns.str.contains('bin'))]]
    kin_calc_bin = combined[combined.columns[combined.columns.str.contains('calc') & combined.columns.str.contains('bin')]]
    kin_arr = [kin_ind, kin_car_cat, kin_calc_not_bin, kin_calc_bin]
    for i, kin in enumerate(kin_arr):
        combined['kin_{}'.format(i+1)] = kin.apply(kinetic, axis=1)
    print ('Transform kinetic features successfully')
    return combined


def cat_transform(combined, type):
    cat_col = combined.columns[combined.columns.str.endswith('cat') == True]
    if type == 'onehot':
        combined = pd.get_dummies(combined, columns=combined.columns[combined.columns.str.endswith('cat')==True])
    elif type == 'count':
        for col in cat_col:
            col_map = combined[col].value_counts()
            combined[col] = combined[col].map(col_map)
    elif type == 'mean':
        train, test = recover_train_test_na(combined, fillna=False)
        target = train.target
        mean_encoder = Bayesian_Encoding(nfolds=5, mode='likelihood')
        encoded_train, encoded_test = mean_encoder.fit_transform(train[cat_col], test[cat_col], target)
        for col in cat_col:
            train[col] = encoded_train[col]
            test[col] = encoded_test[col]
        combined = pd.concat([train.drop('target', axis=1), test], axis=0)
    elif type == 'smooth':
        train, test = recover_train_test_na(combined, fillna=False)
        target = train.target
        smooth_train, smooth_test = target_encode(train[cat_col],
                         test[cat_col],
                         target,
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
        for col in cat_col:
            train[col] = smooth_train[col]
            test[col] = smooth_test[col]
        combined = pd.concat([train.drop('target', axis=1), test], axis=0)

    return combined


def recon_category(combined):
    combined['ps_ind_0609_cat'] = np.zeros_like(combined.ps_ind_06_bin)
    combined['ps_ind_0609_cat'][combined.ps_ind_06_bin==1] = 1
    combined['ps_ind_0609_cat'][combined.ps_ind_07_bin==1] = 2
    combined['ps_ind_0609_cat'][combined.ps_ind_08_bin==1] = 3
    combined['ps_ind_0609_cat'][combined.ps_ind_09_bin==1] = 4
    combined['ps_ind_0609_cat'][combined.ps_ind_0609_cat==0] = 5

    combined.drop(combined.loc[:,'ps_ind_06_bin':'ps_ind_09_bin'].columns, axis=1, inplace=True)

    return combined

def data_transform(combined, type):
    print(type, 'has been selected')
    if type == 'log':
        combined = combined.apply(np.log1p)
    elif type == 'round':
        combined = combined.round(2)
    elif type == 'power':
        combined = combined.apply(lambda x: x**2)
    elif type == 'sqrt':
        combined = combined.apply(np.sqrt)
    elif type == 'minmax':
        scaler = MinMaxScaler()
        combined = scaler.fit_transform(combined)
    elif type == 'std':
        scaler = StandardScaler()
        combined = scaler.fit_transform(combined)
    elif type == 'pca':
        pass
    elif type == 'tsne':
        scaler = StandardScaler()
        scale_combined = scaler.fit_transform(combined)

        tsne = TSNE()
        combined = tsne.fit_transform(scale_combined)
    else:
        print('ERROR NO TYPE !!!')

    return combined

def feature_interactions(combined):
    # Some non-linear features
    combined['ps_car_13_squared'] = (combined.ps_car_13**2*48400).round(0)
    combined['ps_car_12_squared'] = (combined.ps_car_12**2*10000).round(4)
    combined['ps_car_13_x_ps_reg_03'] = combined['ps_car_13'] * combined['ps_reg_03']

    return combined

def engineer_stats(combined):
    combined_car = combined[combined.columns[combined.columns.str.startswith('ps_car') == True]]
    combined_ind = combined[combined.columns[combined.columns.str.startswith('ps_ind') == True]]
    combined_reg = combined[combined.columns[combined.columns.str.startswith('ps_reg') == True]]

    combined['row_na'] = (combined == -1).sum(axis=1)
    combined['count_car_na'] = (combined_car == -1).sum(axis=1)
    combined['count_car_zero'] = (combined_car == 0).sum(axis=1)
    combined['count_car_one'] = (combined_car == 1).sum(axis=1)
    combined['count_ind_na'] = (combined_ind == -1).sum(axis=1)
    combined['count_ind_zero'] = (combined_ind == 0).sum(axis=1)
    combined['count_ind_one'] = (combined_ind == 1).sum(axis=1)
    combined['count_reg_na'] = (combined_reg == -1).sum(axis=1)

    return combined

def recover_train_test_na(combined, fillna='median'):
    if fillna==True:
        # Fill Na
        combined.replace(-1, np.NaN, inplace=True)

    if fillna=='median':
        # Fill Na
        combined.replace(-1, np.NaN, inplace=True)
        combined.fillna(combined.median(), inplace=True)

    # Recover train
    targets = pd.read_csv(DATA_TRAIN_PATH).target.values
    train = combined.iloc[0:595212, :]
    train['target'] = targets
    # Recover test set
    test = combined.iloc[595212:]

    return train, test

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)

class Bayesian_Encoding(object):
    ''' mode can be a str of 'likelihood', 'weight_of_evidence', 'count', 'diff' '''
    def __init__(self, nfolds=5, mode = 'likelihood'):
        self.nfolds = nfolds
        self.mode = mode

    def encoder(self, train, col):
        if self.mode == 'likelihood':
            encoded = train.groupby(col).target.mean()
        elif self.mode == 'weight_of_evidence':
            target = train.groupby(col).target.sum()
            non_target = train.groupby(col).target.count()-target
            encoded = np.log(target/non_target)*100
        elif self.mode == 'count':
            encoded = train.groupby(col).target.sum()
        elif self.mode == 'diff':
            target = train.groupby(col).target.sum()
            non_target = train.groupby(col).target.count()-target
            encoded = target-non_target
        else:
            print('Error!! Please specify encoding mode')
        return encoded

    def cv_encoder(self, train, val, cols):
        for col in cols:
            target_mean = self.encoder(train, col)
            val[col] = val[col].map(target_mean)
        return val

    def global_mean(self, train):
        mean = pd.Series(np.zeros(train.columns.shape), index=train.columns)
        for col in train.columns:
            mean[col] = self.encoder(train, col).mean()
        return mean

    def fit_transform(self, train, test, target):
        train = pd.concat([train, target], axis=1)
        X_train = train.drop('target', axis=1)
        X_test = test
        cols = X_train.columns
        encoded_train = pd.DataFrame(np.zeros(X_train.shape), columns=X_train.columns)
        encoded_test = pd.DataFrame(np.zeros(X_test.shape), columns=X_test.columns)


        skf = StratifiedKFold(n_splits=self.nfolds, shuffle=False)
        for i, (train_index, val_index) in enumerate(skf.split(train, train.target)):
            print('[START ENCODING Fold {}/{}]'.format(i + 1, self.nfolds))
            X_tr, X_val = train.iloc[train_index,:], train.iloc[val_index,:]
            encoded_train.iloc[val_index,:] = self.cv_encoder(X_tr, X_val, cols)
            encoded_test += self.cv_encoder(X_tr, X_test, cols)/5

        # fill NA of encoded using Global Mean
        encoded_train = encoded_train.fillna(self.global_mean(train))
        encoded_test = encoded_test.fillna(self.global_mean(train))
        print('Successfully Encoded')
        return encoded_train, encoded_test

def make_submission(pred, filename):
    p = pd.read_csv('oof/'+ str(pred)).values.reshape(892816,)
    sub_id = pd.read_csv('data/test.csv').id
    sub = pd.DataFrame({'id':sub_id, 'target':p})
    sub.to_csv('submissions/{}'.format(filename), index=False)
    print('Create {} successfully'.format(filename))


def bojan_engineer(df):
    # from olivier
    features = ["ps_car_13", "ps_reg_03", "ps_ind_05_cat", "ps_ind_03", "ps_ind_15", "ps_reg_02", "ps_car_14", "ps_car_12", "ps_car_01_cat", 
    "ps_car_07_cat","ps_ind_17_bin", "ps_car_03_cat", "ps_reg_01", "ps_car_15", "ps_ind_01", "ps_ind_16_bin", "ps_ind_07_bin", "ps_car_06_cat", 
    "ps_car_04_cat", "ps_ind_06_bin", "ps_car_09_cat", "ps_car_02_cat", "ps_ind_02_cat", "ps_car_11", "ps_car_05_cat", "ps_calc_09", "ps_calc_05", 
    "ps_ind_08_bin", "ps_car_08_cat", "ps_ind_09_bin", "ps_ind_04_cat", "ps_ind_18_bin", "ps_ind_12_bin", "ps_ind_14"]

    # add combinations
    combs = [('ps_reg_01', 'ps_car_02_cat'),  ('ps_reg_01', 'ps_car_04_cat')]

    df = df[features]

    for f1, f2 in combs:
        name = f1 + "_plus_" + f2
        df[name] = df[f1].apply(lambda x: str(x)) + "_" + df[f2].apply(lambda x: str(x))
        lbl = LabelEncoder()
        lbl.fit(list(df[name].values))
        df[name] = lbl.transform(list(df[name].values))

    return df

def fold_smoothing_encode(X_train, X_val, X_test, y_train):
    for f in X_train.columns[X_train.columns.str.endswith('cat')==True]:
        X_train[f + "_avg"], X_val[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_val[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    return X_train, X_val, X_test

def minmaxpandas(df):
    for col in df.columns:
        maxcol = df[col].max()
        mincol = df[col].min()
        df[col] = df[col].apply(lambda x: (x-mincol)/(maxcol-mincol))
    return df