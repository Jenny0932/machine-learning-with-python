import pandas as pd
import quandl, math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
df_original         = quandl.get('WIKI/GOOGL')
df                  = quandl.get('WIKI/GOOGL')
df                  = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT']        = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0
df['PCT_change']    = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0
df                  = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col        = 'Adj. Close'
REPLACE_NA          = -9999
df.fillna(REPLACE_NA, inplace = True)
PREDICTION_RATE     = 0.01 
forecast_out        =  int(math.ceil(PREDICTION_RATE*len(df)))
df['label']         =  df[forecast_col].shift(-forecast_out)
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
confidence = clf.score(X_test,y_test)
print(confidence)

with open('linearregression_googleWIKI.pickle', 'wb') as f:
    pickle.dump(clf, f)


pickle_in = open('linearregression_googleWIKI.pickle', 'rb')
clf = pickle.load(pickle_in)
forecast_set=clf.predict(X_lately)
forecast_set=np.array(forecast_set)
df['Forecast'] = np.nan
last_date=df_original.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day=86400
next_unix=last_unix+one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)

    #print(next_date)
    #print(i)

    next_unix += one_day
    #.loc refers the index
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]



df_original['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()