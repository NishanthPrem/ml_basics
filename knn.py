#%% Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Loading and splitting the dataset

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y , test_size=0.25, random_state=0)

#%% Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Initializing the classifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

#%% Prediction

y_pred = classifier.predict(X_test)

#%% Confusion Matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

