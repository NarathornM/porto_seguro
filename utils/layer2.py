import glob
from constants import *
import pandas as pd

def get_meta_features(mode):
    # m = observations n = numbers of models
    if mode == 'train':
        # List of csv path
        csv_list = glob.glob(OOF_PATH + '/*_train.csv')
        # Base meta features dataframe
        meta_features_df = pd.read_csv(csv_list[0])

        for f in csv_list[1:]:
            tmp_df = pd.read_csv(f)
            meta_features_df = pd.concat([meta_features_df, tmp_df], axis=1)

        return meta_features_df

    elif mode == 'test':
        # List of csv path
        csv_list = glob.glob(OOF_PATH + '/*_test.csv')

        meta_features_df = pd.read_csv(csv_list[0])

        for f in csv_list[1:]:
            tmp_df = pd.read_csv(f)
            meta_features_df = pd.concat([meta_features_df, tmp_df], axis=1)

        return meta_features_df

    else:
        print('ERROR ! NO MODE FOUND!!!')