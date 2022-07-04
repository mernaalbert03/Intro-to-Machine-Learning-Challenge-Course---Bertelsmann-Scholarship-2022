'''
L1 Regularization 
'''
import numpy as np 
import pandas as pd 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import LinearRegression

#1)load in the data 
train_data=pd.read_csv('data.csv')
x=train_data.iloc[:,:-1].values
y=train_data.iloc[:,-1].values

#Linear Regression without regularization

model=LinearRegression()
model.fit(x,y)

#2)creat and fit data using Linear regression with lasso regularization 
lasso_reg=Lasso()
lasso_reg.fit(x,y)

linear_model=model.coef_   #coefficients before regularization

#3)Retrieve and print out the coefficients from the regression model with regularization
reg_coef=lasso_reg.coef_


print(linear_model)  #coefficients without regularization

#result
#[-6.37721775e-03  2.96142992e+00  1.98263996e+00 -7.87140186e-02
# -3.95756190e+00  9.28865637e+00]

print(reg_coef)      #coefficients with regularization

#result
#[ 0.          2.33659619  2.0140086  -0.05753445 -3.91583673  0.        ]

#note:.coef_ attribute of lasso object store the weights(coeffficents of the fit regression model) in the reg_coef variable 

