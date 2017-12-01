
# coding: utf-8

# https://www.kaggle.com/tilii7/keras-averaging-runs-gini-early-stopping


from utils import utils, gini
import time
from constants import *
import os
import gc

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from keras.models import load_model, Sequential
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Concatenate, Merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding



np.random.seed(88) # for reproducibility
MODEL_NAME = 'keras_smooth_itt'
SEED = 88

combined = utils.load_data()
# combined = utils.bojan_engineer(combined)
combined = utils.drop_stupid(combined)
combined = utils.engineer_stats(combined)
combined = utils.recon_category(combined)
combined = utils.minmaxpandas(combined)
combined = combined.replace(np.NaN, -1)
# combined = utils.cat_transform(combined, 'onehot')
# combined = utils.data_transform(combined, self.data_transform)
# combined = utils.feature_interactions(combined)
train, test = utils.recover_train_test_na(combined, fillna=False)

X_train = train.drop('target', axis=1)
y_train = train.target
# X_test = test


print('\n')

class gini_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.X_tr = training_data[0]
        self.y_tr = training_data[1]
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_lap = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_tr = self.model.predict_proba(self.X_tr)
        logs['gini_tr'] = gini.gini_sklearn(self.y_tr, y_pred_tr)
        y_pred_val = self.model.predict_proba(self.X_val)
        logs['gini_val'] = gini.gini_sklearn(self.y_val, y_pred_val)

        # if logs['gini_val'] > self.best_lap:
        #     self.best_lap = logs['gini_val']

        #     global pred_val, pred_test
        #     pred_val = y_pred_val
        #     pred_test = self.model.predict_proba(X_test)

        print('Gini Score in training set: {},  test set: {}'.format(logs['gini_tr'], logs['gini_val']))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test


def create_model():

    model = Sequential()
    model.add(Dense(80, input_dim=X_tr.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model




epochs = 1024
batch_size = 4096
patience = 10
KFOLDS = 5
runs_per_fold = 3


tmp = time.time()
skf = StratifiedKFold(n_splits=KFOLDS, random_state=SEED)
scores = []
oof_train = np.zeros((X_train.shape[0],))
oof_test = np.zeros((test.shape[0],))


for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    assert len(X_train) == len(y_train)

    score_fold = []

    print('\n')

    print('[Fold {}/{} START]'.format(i + 1, KFOLDS))

    X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
    y_tr, y_val = y_train[train_index], y_train[val_index]

    
    # smoothing
    X_tr, X_val, X_test = utils.fold_smoothing_encode(X_tr, X_val, test, y_tr)

    X_tr = X_tr.values
    X_val = X_val.values
    X_test = X_test.values


    for j in range(runs_per_fold):
        print('Starting run {}'.format(j+1))

        pred_val = np.zeros((len(val_index),1))
        pred_test = np.zeros((892816,))
        log_path = os.path.join(LOG_PATH, MODEL_NAME + '_log.csv')
        checkpoint_path = os.path.join(LOG_PATH, MODEL_NAME + '_check.check'.format(j))

        callbacks = [
        gini_callback(training_data=(X_tr, y_tr), validation_data=(X_val, y_val)),
        EarlyStopping(monitor='gini_val', patience=patience, mode='max', verbose=1),
        CSVLogger(log_path, separator=',', append=False),
        ModelCheckpoint(checkpoint_path, monitor='gini_val', mode='max', save_best_only=True, save_weights_only=True, verbose=1)
        ]

        model = create_model()


        model.fit(X_tr, y_tr, shuffle=False, batch_size=batch_size, epochs=epochs, verbose=99, callbacks=callbacks)

        # delete current model
        del model

        # load best model of each run
        model = create_model()
        model.load_weights(checkpoint_path, by_name=False)

        # For train and valid only
        pred_val = model.predict_proba(X_val).reshape(-1, )

        # Store average score for evaluate model
        score_fold.append(gini.gini_sklearn(y_val, pred_val))

        oof_train[val_index] += pred_val / runs_per_fold

        print('Run {}: {} \n'.format(j+1, score_fold[j]))

        pred_test_lap = model.predict_proba(X_test).reshape(-1, )
        pred_test += pred_test_lap / runs_per_fold

    # Store test predictions for submissions
    
    oof_test += pred_test / KFOLDS

    scores.append(gini.gini_sklearn(y_val, oof_train[val_index]))
    print('[Fold {}/{} Gini score: {} \n]'.format(i+1, KFOLDS, scores[i]))

    gc.collect()
    print('[Fold {}/{} END \n]'.format(i+1, KFOLDS))

print('Average score: {}'.format(np.mean(scores)))
print('Total run time: {} seconds'.format(time.time() - tmp))

# Export oof_train
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_train.csv')
pd.DataFrame({MODEL_NAME: oof_train}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_train.reshape(-1, 1), delimiter=',', fmt='%.5f')

# Export oof_test
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
pd.DataFrame({MODEL_NAME: oof_test}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_test.reshape(-1, 1), delimiter=',', fmt='%.5f')
print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))
