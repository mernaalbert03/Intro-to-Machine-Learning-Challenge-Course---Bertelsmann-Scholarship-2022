'''
Polynomial Regression 
'''
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

#1)load in the data 
train_data=pd.read_csv('data.csv')

#2)split the data into X (predictor feature) and Y(putcome feature)
#.reshape(-1,1) to transform 1D array to 2D array

x=train_data['Var_X'].values.reshape(-1,1)
y=train_data['Var_Y'].values

#3)create polynomial features 

poly_feat=PolynomialFeatures(degree=4)
x_poly=poly_feat.fit_transform(x)



#4)build a polynomial regression model
#Create a LinearRegression object and fit it to the polynomial predictor features
poly_model = LinearRegression(fit_intercept = False)
poly_model.fit(x_poly,y)








 