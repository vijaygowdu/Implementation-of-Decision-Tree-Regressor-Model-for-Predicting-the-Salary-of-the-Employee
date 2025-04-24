# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: k vijay
RegisterNumber: 212223040236
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:
## data.head()
![Screenshot 2024-04-02 134650](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/497a662d-0254-4501-9da1-f83acb83b672)
## data.info()
![Screenshot 2024-04-02 134714](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/88157a23-fdfa-446d-85a4-75da500b79c2)
## data.isnull().sum()
![Screenshot 2024-04-02 134719](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/11401fc5-4fa3-488c-a591-c181f44aeb94)
## data.head() for salary
![Screenshot 2024-04-02 134727](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/2124bb42-2cd5-4907-b6b2-0f2dd2fd16b9)
## MSE value
![Screenshot 2024-04-02 134744](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/b875ed87-ded1-4f25-8a1a-2cd3b6029904)
## r2 value
![Screenshot 2024-04-02 134749](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/a5cb3052-4f18-463a-a020-ea6e175b017f)
## data prediction
![Screenshot 2024-04-02 134819](https://github.com/rajalakshmi8248/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122860827/b2e3c327-fa8d-4fca-8f67-7be8d6bfb59b)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
