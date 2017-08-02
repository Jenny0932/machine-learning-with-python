import pandas as pd
import quandl, math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')
df_original = quandl.get("SSE/ERCA", authtoken="yHT6Cb3WYsrDN5dwWw2z")
df = quandl.get("SSE/ERCA", authtoken="yHT6Cb3WYsrDN5dwWw2z")
#print(df)
df = df[['High','Low','Last','Previous Day Price','Volume']]
df['HL_PCT'] = (df['High'] - df['Last']) / df['Last']*100.0
df['PCT_change'] = (df['Last'] - df['Previous Day Price']) / df['Previous Day Price']*100.0

# how to select features
df = df[['Last', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Last'
df.fillna(-9999, inplace = True)


# create a ground truth label
forecast_out =  int(math.ceil(0.05*len(df)))


# delete the oldest data
df['label'] =  df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)



X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

# delete the latest
X = X[:-forecast_out]

y = np.array(df['label'])
y = y[:-forecast_out]

#print(df.head())
#print(len(X))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# different machine learning Algorithms
clf = LinearRegression(n_jobs = -1)
#clf = svm.SVR()
#clf = svm.SVR(kernel='poly')
clf.fit(X_train,y_train)
# finish training
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test,y_test)
print(confidence)


forecast_set=clf.predict(X_lately)
forecast_set=np.array(forecast_set)
df['Forecast'] = np.nan


last_date=df_original.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
print(datetime.datetime.fromtimestamp(last_unix))
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
 next_date = datetime.datetime.fromtimestamp(next_unix)
 #print(next_date)
 #print(i)
 next_unix += one_day
 df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
 
 
 
df['Forecast'].plot()
df_original['Last'].plot()

plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()