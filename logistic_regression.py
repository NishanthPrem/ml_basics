#%% Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Loading and identifying the dependent and independent variables

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the data into train and test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

#%% Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Applying Logistic Regression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#%% Prediction

prediction = classifier.predict(sc.transform([[30,87000]]))

#%% Predicting X_test

y_pred = classifier.predict(X_test)

#%% Confusion Matrix for accuracy

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)