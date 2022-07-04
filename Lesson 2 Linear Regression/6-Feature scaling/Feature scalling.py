'''
Feature scaling 

'''

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso 

#1) load in the data 
train_data=pd.read_csv('data.csv')
x=train_data.iloc[:,:-1].values
y=train_data.iloc[:,-1].values

#2)preform feature scalling on data via standardization 
scaler=StandardScaler()
#3) compute the scalling parameter on the predictor feature array 
x_scaled=scaler.fit_transform(x)

#4)fit data using linear regression with lasso regularization 
lasso_reg=Lasso()
lasso_reg.fit(x_scaled,y)
reg_coef=lasso_reg.coef_
print(reg_coef)

#result
#reg_coef=[  0.     3.8596924    9.05021225  -0.      -11.72692976       0.41040086]            