from utils import layer2
from constants import *

import pandas as pd

# Sequential model type from Keras. Good for building feed-forward nn like this
from keras.models import Sequential
# Next, let's import the "core" layers from Keras. These are the layers that are used in almost any neural network
from keras.layers import Dense, Dropout
# K-fold-validation
from sklearn.model_selection import StratifiedKFold

kfolds = 5
np.random.seed(22)

def main():
    train = layer2.get_meta_features('train')
    test = layer2.get_meta_features('test')

    X_train = train.values
    y_train = pd.read_csv(DATA_TRAIN_PATH).target.values
    X_test = test.values

    skf = StratifiedKFold(n_splits=kfolds, random_state=22)

    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model.add(Dense(64, input_dim=X_tr.shape[0], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(x_tr, y_tr,
                  epochs=20,
                  batch_size=128)
        score = model.evaluate(x_test, y_test, batch_size=128)





if __name__ == '__main__':
    main()
