import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Rainfall.csv', index_col=0,
     parse_dates=True, infer_datetime_format=True)

df.head()

print(df.dtypes)

first_date = df.index[0]
first_date

df.Rainfall.head()

df.plot()

RECENT_PERIOD = 5  
df.Rainfall[-RECENT_PERIOD:]

df_recent_period = df[-RECENT_PERIOD:] 

df_recent_period.head()

df_recent_period.Rainfall.mean()  

df_recent_period.Rainfall.sum()

rolling_mean_5 = df.Rainfall.rolling(window=5).mean().shift(1)  

rolling_mean_12 = df.Rainfall.rolling(window=12).mean().shift(1)

print(df.Rainfall.head())
print("--  rolling mean 5")
print(rolling_mean_5.head())
print("--  rolling mean 12")
print(rolling_mean_12.head())

plt.plot(df.index, df.Rainfall, label='Rainfall')
plt.plot(df.index, rolling_mean_5, label='5 Months SMA', color='orange')
plt.plot(df.index, rolling_mean_12, label='12 Months SMA', color='magenta')
plt.legend(loc='upper left')
plt.show()

print("5")

print(rolling_mean_5[10])

rolling_mean_5.to_pickle("moving_average2.pkl")
