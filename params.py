 # 0.2852143 #fillmed 0.2855303 #5f med 0.2867438
params_xgb_depth_5 = {}
params_xgb_depth_5['tree_method']= 'hist'
params_xgb_depth_5['silent']= True 
params_xgb_depth_5['eval_metric']= 'auc'
params_xgb_depth_5['objective']= 'binary:logistic'
params_xgb_depth_5['eta']= 0.01
params_xgb_depth_5['max_depth']= 5
params_xgb_depth_5['gamma']= 0.3
params_xgb_depth_5['min_child_weight']= 7
params_xgb_depth_5['subsample']= 0.7 
params_xgb_depth_5['colsample_bytree']= 0.6
params_xgb_depth_5['max_delta_step']= 5


#xgb_10_rep
params_xgb_depth_10 = {} 
params_xgb_depth_10['tree_method']= 'hist' 
params_xgb_depth_10['silent']= True 
params_xgb_depth_10['eval_metric']= 'auc' 
params_xgb_depth_10['objective']= 'binary:logistic' 
params_xgb_depth_10['eta']= 0.01 
params_xgb_depth_10['max_depth']= 10 
params_xgb_depth_10['gamma']= 5 
params_xgb_depth_10['min_child_weight']= 9 
params_xgb_depth_10['subsample']= 0.7 
params_xgb_depth_10['colsample_bytree']= 0.6 
params_xgb_depth_10['max_delta_step']= 0.1 
params_xgb_depth_10['scale_pos_weight']= 10
 #0.284297 dropstupid onehot
 #0.285098 target encode, some calc, engineer stat LB:0.281
# params_xgb_depth_10 = {}
# params_xgb_depth_10['tree_method']= 'hist'
# params_xgb_depth_10['silent']= True
# params_xgb_depth_10['eval_metric']= 'auc'
# params_xgb_depth_10['objective']= 'binary:logistic'
# params_xgb_depth_10['eta']= 0.01
# params_xgb_depth_10['max_depth']= 5
# params_xgb_depth_10['gamma']= 0.3
# params_xgb_depth_10['min_child_weight']= 9
# params_xgb_depth_10['subsample']= 0.7
# params_xgb_depth_10['colsample_bytree']= 0.6
# params_xgb_depth_10['max_delta_step']= 5
# params_xgb_depth_10['scale_pos_weight']= 10

params_lgb_depth_6 = {}
params_lgb_depth_6['verbosity'] = -1
params_lgb_depth_6['metric'] = 'auc'
params_lgb_depth_6['objective'] = 'binary'
params_lgb_depth_6['learning_rate'] = 0.01
params_lgb_depth_6['min_split_gain'] = 0.1
params_lgb_depth_6['min_child_weight'] = 7
params_lgb_depth_6['subsample'] = 0.5
params_lgb_depth_6['colsample_bytree'] = 0.6
params_lgb_depth_6['poission_max_delta_step'] = 5
params_lgb_depth_6['bagging_fraction'] = 0.85
params_lgb_depth_6['bagging_freq'] = 40
params_lgb_depth_6['num_leaves'] = 1024
params_lgb_depth_6['min_data'] = 500
params_lgb_depth_6['lambda_l1'] = 16.7 



#0.2833283
params_lgb_depth_10 = {}
params_lgb_depth_10['verbosity'] = -1
params_lgb_depth_10['metric'] = 'auc'
params_lgb_depth_10['objective'] = 'binary'
params_lgb_depth_10['learning_rate'] = 0.01
params_lgb_depth_10['max_depth'] = 10
params_lgb_depth_10['min_split_gain'] = 0.1
params_lgb_depth_10['min_child_weight'] = 7
params_lgb_depth_10['poission_max_delta_step'] = 0.3
params_lgb_depth_10['subsample'] = 0.5
params_lgb_depth_10['colsample_bytree'] = 0.6

params_lgb = {}
params_lgb['verbosity'] = -1
params_lgb['metric'] = 'auc'
params_lgb['objective'] = 'binary'
params_lgb['learning_rate'] = 0.01
params_lgb['max_depth'] = 10
params_lgb['min_split_gain'] = 1
params_lgb['min_child_weight'] = 7
params_lgb['subsample'] = 0.5
params_lgb['colsample_bytree'] = 0.6


params_rgf = {}
params_rgf['max_leaf'] = 1000
params_rgf['algorithm'] = 'RGF'
params_rgf['loss'] = 'Log'
params_rgf['l2'] = 0.01
params_rgf['sl2'] = 0.01
params_rgf['normalize'] =False
params_rgf['min_samples_leaf'] = 10
params_rgf['n_iter'] = None
params_rgf['opt_interval'] = 100
params_rgf['learning_rate'] = .5
params_rgf['calc_prob'] = 'sigmoid'
params_rgf['n_jobs'] = -1
params_rgf['memory_policy'] = 'generous'
params_rgf['verbose'] = 0


params_catboost = {}
params_catboost['learning_rate'] = 0.05 
params_catboost['depth'] = 6
params_catboost['l2_leaf_reg'] = 14
params_catboost['iterations'] = 650
params_catboost['verbose'] = False
params_catboost['loss_function'] ='Logloss'


lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['num_round'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99


lgb_params2 = {}
lgb_params2['num_round'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99


lgb_params3 = {}
lgb_params3['verbosity'] = -1
lgb_params3['num_round'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99

lgb_params4 = {}
lgb_params4['verbosity'] = -1
lgb_params4['num_round'] = 1450
lgb_params4['max_bin'] = 20
lgb_params4['max_depth'] = 6
lgb_params4['learning_rate'] = 0.02 # shrinkage_rate
lgb_params4['boosting_type'] = 'gbdt'
lgb_params4['objective'] = 'binary'
lgb_params4['min_data'] = 500         # min_data_in_leaf
lgb_params4['min_hessian'] = 0.05     # min_sum_hessian_in_leaf