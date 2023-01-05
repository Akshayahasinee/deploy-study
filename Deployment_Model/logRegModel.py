import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read data file
df = pd.read_csv('gender_data.csv')

# split dataframe to two dataframes male and female
# for scatter plot
# follow column name in file
df0 = df[df['Sex'] == 0]
df1 = df[df['Sex'] == 1]

# model training
logReg = LogisticRegression(solver = 'lbfgs')

x = df[['Height', 'Weight']]
y = df['Sex'] # Classification : Sex = 0 or 1 (Male or Female)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

logReg.fit(x_train, y_train)

# deploy to the flask server
# flask server need to be started
pickle.dump(logReg, open('logRegModel.pkl', 'wb'))  #serialize the object