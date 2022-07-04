'''
Linear Regression in scikit-learn 
use linear reagression to predict life expectary from body mass index(BMI)
'''
#1)load the data

from sklearn.linear_model import LinearRegression 
import pandas as pd
bmi_life_data=pd.read_csv('bmi_and_life_expectancy.csv')

#2)build a linear regression model 
x=bmi_life_data[['BMI']]
y=bmi_life_data[['Life expectancy']]
bmi_life_model=LinearRegression()
bmi_life_model.fit(x,y)

#note: y_hat=model.predict(x)
plt.scatter(x,y)
plt.plot(x, bmi_life_model.predict(x),color='k')
plt.show()

#3)predict using model 
loas_life_exp=bmi_life_model.predict([[21.07931]])

#note:  .predict the data must be in a form of 2D array  

#result=60.3156
