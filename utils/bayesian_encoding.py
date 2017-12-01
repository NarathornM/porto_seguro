import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

    def fit_transform(self, train, test):        
        X_train = train.drop(['id', 'target'], axis=1)
        X_test = test.drop('id', axis=1)
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