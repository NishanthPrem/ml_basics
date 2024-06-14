#%% Importing Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Loading the dataset

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values

#%% Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

#%% Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% SVM

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

#%% Prediction

y_pred = classifier.predict(X_test)

#%% Confusion Matrix and Accuracy score

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)