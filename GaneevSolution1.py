import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train=pd.read_csv('sample_data/train.csv',encoding='utf-8')
test=pd.read_csv('sample_data/test.csv',encoding='utf-8')

y = train.per_square_meter_price

data_predictors = list(train.dtypes[train.dtypes == "int64"].index)
data_predictors = data_predictors[:-1]
data_predictors

data_test_predictors = list(test.dtypes[train.dtypes == "int64"].index)
data_test_predictors = data_test_predictors[:-1]
data_test_predictors

X = train[data_predictors]

X_test = test[data_test_predictors]

model = DecisionTreeRegressor()

model.fit(X,y)

predict_test = model.predict(X_test)

test['per_square_meter_price'] = predict_test

test[['id','per_square_meter_price']].to_csv('subGaneev.csv',header=True,index=False)