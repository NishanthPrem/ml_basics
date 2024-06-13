#%% Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% Loading and identifying the dependent and independent variables

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)