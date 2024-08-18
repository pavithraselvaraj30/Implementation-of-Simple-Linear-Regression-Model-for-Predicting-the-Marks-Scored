## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn
2. Calculate the values for the training data set
3. Calculate the values for the test data set
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pavithra S
RegisterNumber: 212223230147
/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

##  splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

## graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![Screenshot 2024-08-18 223903](https://github.com/user-attachments/assets/0df95980-c22f-4a5e-b8f8-04700b19bc5b)
![Screenshot 2024-08-18 223255](https://github.com/user-attachments/assets/93176809-5184-411c-ac8d-00f2ceddcc2c)
![Screenshot 2024-08-18 223313](https://github.com/user-attachments/assets/bfd28c6c-6d76-4dfb-a260-091f009989ff)
![Screenshot 2024-08-18 223633](https://github.com/user-attachments/assets/dbd1c6d1-dc76-42c8-ad11-7d7d20152da3)
![Screenshot 2024-08-18 223648](https://github.com/user-attachments/assets/e96df2fe-ebac-42f4-81dc-d3e9da0e6714)
![Screenshot 2024-08-18 223704](https://github.com/user-attachments/assets/0c0dfd27-21a3-4015-97c0-fc2c64ecde50)
![Screenshot 2024-08-18 223721](https://github.com/user-attachments/assets/4a83a95b-757d-4161-a7b0-301d5fc4fcaa)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
