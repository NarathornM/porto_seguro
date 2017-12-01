from params import *
from utils import layer1, utils
from rgf.sklearn import RGFClassifier
from catboost import CatBoostClassifier



for i in [100,200,300,400,500]:


	print(params_xgb_depth_10)
	xgb_1 = layer1.Layer1Train('xgb', params_xgb_depth_10, 'xgb_depth_10_seed{}'.format(i), cv='kf', fillna='median', my_features=False, 
			drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
			feature_interactions=False, engineer_stats=False, kinetic_feature=False, seed=i, shuffle=False)

	xgb_1.train()


	print(params_xgb_depth_5)
	xgb_1 = layer1.Layer1Train('xgb', params_xgb_depth_5, 'xgb_5kf_seed{}'.format(i), cv='kf', fillna='median', my_features=False, 
		drop_stupid=True, cat_transform='onehot', data_transform=False, recon_category=False, 
		feature_interactions=False, engineer_stats=False, kinetic_feature=False, seed=i, shuffle=True)

	xgb_1.train()

	# print(lgb_params)
	lgb_1 = layer1.Layer1Train('lgb', lgb_params, 'lgb_harless1_itt', cv='skf', fillna=True, my_features=False, 
		smoothing=True, drop_stupid=True, cat_transform=False, data_transform=False, recon_category=True, 
		feature_interactions=False, engineer_stats=True, kinetic_feature=False)

	lgb_1.train()


	print(params_rgf)
	rgf_1 = layer1.Layer1Train(RGFClassifier(**params_rgf),'' , 'rgf_seed{}'.format(i), cv='kf', fillna=False, my_features=False, 
		smoothing=True, drop_stupid=True, cat_transform=False, recon_category=True, feature_interactions=False, engineer_stats=True, seed=i, shuffle=True)

	rgf_1.train()