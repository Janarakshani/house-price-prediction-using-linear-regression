

import pandas as jana
datagreen=jana.read_csv(r'/content/housing-3.csv')

datagreen

datagreen.info()

datagreen_x=datagreen.drop(['RAD'],axis=1)

datagreen_y=datagreen['RAD']

datagreen_x

datagreen_y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split (datagreen_x,datagreen_y,test_size=0.25,random_state=42)

from sklearn.linear_model import LinearRegression

LR=LinearRegression()

LR.fit(x_train,y_train)

y_predicted=LR.predict(x_test)

y_predicted

from sklearn.metrics import mean_squared_error

LR.score(x_test,y_test)

mse_lr_model=mean_squared_error(y_test,y_predicted)

mse_lr_model

import math
rmse=math.sqrt(mse_lr_model)

rmse

import numpy as jana

y=datagreen['RAD'].values
mean_y=jana.mean(y)
squared_diff=(y - mean_y)**2
variance_y=jana.mean(squared_diff)
print("Mean:", mean_y)
print("Variance:", variance_y)

datagreen.describe()

mean_target_variable=9.549407114624506
normalized_mse = mse_lr_model / mean_target_variable
normalized_variance =variance_y / mean_target_variable

print("Normalized MSE:", normalized_mse)
print("Normalized Variable:", normalized_variance)
