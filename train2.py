from params import *
from utils import layer1, utils



print(params_xgb_depth_10)
xgb_1 = layer1.Layer1Train('xgb', params_xgb_depth_10, 'xgb_depth_10_rep', cv='kf', fillna='median', my_features=False, 
		drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
		feature_interactions=False, engineer_stats=False, kinetic_feature=False, seed=88, shuffle=False)

xgb_1.train()

# print(lgb_params)
# lgb_1 = layer1.Layer1Train('lgb', lgb_params, 'lgb_harless1_itt', cv='skf', fillna=True, my_features=False, 
# 	smoothing=True, drop_stupid=True, cat_transform=False, data_transform=False, recon_category=True, 
# 	feature_interactions=False, engineer_stats=True, kinetic_feature=False)

# lgb_1.train()

# print(lgb_params2)
# lgb_1 = layer1.Layer1Train('lgb', lgb_params2, 'lgb_harless2_calckf', cv='kf', fillna=True, my_features=False, 
# 	bojan_features=False, drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
# 	feature_interactions=False, engineer_stats=False, kinetic_feature=False)

# lgb_1.train()

# print(lgb_params3)
# lgb_1 = layer1.Layer1Train('lgb', lgb_params3, 'lgb_harless3_calckf', cv='kf', fillna=True, my_features=False, 
# 	bojan_features=False, drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
# 	feature_interactions=False, engineer_stats=False, kinetic_feature=False)

# lgb_1.train()

# print(lgb_params4)
# lgb_1 = layer1.Layer1Train('lgb', lgb_params4, 'lgb_harless4_calckf', cv='kf', fillna=True, my_features=False, 
# 	bojan_features=False, drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
# 	feature_interactions=False, engineer_stats=False, kinetic_feature=False)

# lgb_1.train()


# utils.make_submission('xgb_homeless_test.csv', 'sub_xgb_homeless.csv')
# utils.make_submission('xgb_5kf_test.csv', 'sub_xgb_5kf.csv')
# utils.make_submission('lgb_harless1_calckf_test.csv', 'sub_lgb_harless1_calckf.csv')
utils.make_submission('keras_itt_seed88_test.csv', 'sub_keras_itt_seed88.csv')
# 	