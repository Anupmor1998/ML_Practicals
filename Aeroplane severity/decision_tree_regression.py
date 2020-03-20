# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
X = train_dataset.iloc[:, 1:].values
y = train_dataset.iloc[:, 0:1].values
test_dataset=pd.read_csv('test.csv')
X_test = test_dataset.values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y=LabelEncoder()
y[:,0]=labelencoder_y.fit_transform(y[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(y).toarray()

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred=regressor.predict(X_test)

#Converting the output to csv file
decoded = y_pred.dot(onehotencoder.active_features_).astype(int)
decode1=labelencoder_y.inverse_transform(decoded)
result=pd.DataFrame(decode1)
result.to_csv('prediction.csv')
