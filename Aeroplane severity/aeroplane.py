#Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
data = pd.read_csv('test.csv')
X1 = data.values
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
y_label = LabelEncoder()
y[:,0] = y_label.fit_transform(y[:,0])
onehotencoder = OneHotEncoder() 
y = onehotencoder.fit_transform(y).toarray() 



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Predicting a new result
for i in range(2500):
    y_pred = regressor.predict([X1[i]])

    inverted = y_label.inverse_transform([np.argmax(y_pred)])
    prediction = pd.DataFrame(inverted,columns=['prediction']).to_csv('prediction.csv',mode='a',header=False)
    

