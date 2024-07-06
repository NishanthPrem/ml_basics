#%% Importing the libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Importing the dataset

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the dataset

X_train, X_test, y_train, y_true = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%% Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Naive Bayes

classifier = GaussianNB()
classifier.fit(X_train, y_train)

#%% Prediction

y_pred = classifier.predict(X_test)

#%% Evaluating the model

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)