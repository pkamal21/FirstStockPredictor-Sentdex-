import math, datetime
import pandas as pd
import quandl as q
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
style.use("ggplot")
q.ApiConfig.api_key = "u7U8X8jZ_QfZfVv7HZBx"

df = q.get('NSE/TCS')

df = df[['Open','High', 'Low', 'Close', 'Total Trade Quantity']]
df['HL_PCT'] = (df['High']-df['Close'])/df['Close']*100
df['PCT_change'] = (df['Close']-df['Open'])/df['Open']*100

df = df[['Close', 'HL_PCT', 'PCT_change', 'Total Trade Quantity']]

forecast_col = 'Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately  =X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
