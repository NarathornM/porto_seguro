https://github.com/ahara/kaggle_otto/blob/master/otto
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python?scriptVersionId=898172

python -m models.[name]
for example python -m models.model_02_knn_2


# Apply Log transformation log(x+1)
  # train[train.drop(['id', 'target'], axis=1).columns] = train.drop(['id', 'target'], axis=1).apply(np.log1p)
  # test[test.drop('id', axis=1).columns] = test.drop('id', axis=1).apply(np.log1p)
