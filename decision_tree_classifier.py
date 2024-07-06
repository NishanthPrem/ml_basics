#%% Importing the libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Importing the dataset

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%% Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Decision Tree

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#%% Prediction

y_pred = classifier.predict(X_test)

#%% Evaluating the model

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)