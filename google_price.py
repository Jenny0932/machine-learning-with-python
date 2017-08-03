import pandas as pd
import numpy as np
import quandl
import math, datetime, time
from sklearn import preprocessing,cross_validation,svm,linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = 'yHT6Cb3WYsrDN5dwWw2z'
# change ticker and date to filter the data: ticker = 'FB' for facebook, ticker = 'GOOGL' for google
df          = quandl.get_table('WIKI/PRICES', date = { 'gte': '1990-01-01', 'lte': '2017-08-01' }, ticker = 'GOOGL')
df_original = quandl.get_table('WIKI/PRICES', date = { 'gte': '1990-01-01', 'lte': '2017-08-01' }, ticker = 'GOOGL')

# prepare feature data 
df['HL_PCT']        =(df['adj_high']-df['adj_close'])/df['adj_close'] * 100.0
df['PCT_change']    =(df['adj_close']-df['adj_open']) / df['adj_open'] * 100.0
df                  = df[['adj_open','adj_high','adj_low','adj_close','HL_PCT','PCT_change','adj_volume','date']]
df.fillna(-9999, inplace = True)
X_tmp               =np.array(df.drop(['date'],1))
df                  = pd.DataFrame(X_tmp, index=df['date'], columns = ['adj_open','adj_high','adj_low','adj_close','HL_PCT','PCT_change','adj_volume'])
df.sort_index(inplace=True)

# prepare original data for show and find the latest date
df_original = df_original[['adj_close','adj_volume','date']]
df_original.fillna(-9999, inplace = True)
X_original  = np.array(df_original.drop(['date'],1))
df_original = pd.DataFrame(X_original, index=df_original['date'], columns = ['adj_close','adj_volume'])
df_original.sort_index(inplace=True)



# remove testing data from original data
forecast_col    = df['adj_close']
prediction_rate = 0.001 
forecast_out    = int(math.ceil(prediction_rate*len(forecast_col)))
df['label']     = forecast_col.shift(-forecast_out)
df.dropna(inplace=True)
print "We predict for next ", forecast_out, " days"


X    =np.array(df.drop(['label'],1))#,'adj_close'],1))
X    =preprocessing.scale(X)
# get the last few elements in X to create a prediction
# generate testing data
X_lately = X[-forecast_out:]
# generate training data
X    = X[:-forecast_out]
Y    = np.array(df['label'])
Y    = Y[:-forecast_out]

# training model
testing_rate = 0.2 # 20% used for testing
x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,Y, test_size=testing_rate)

clf      =linear_model.LinearRegression(n_jobs=-1)
#clf = svm.SVR()
#clf = svm.SVR(kernel='poly')
clf      = clf.fit(x_train,y_train)
accuracy =clf.score(x_test,y_test)*100
accuracy = "%.2f" % accuracy
print "Accuracy for cross_validation: ", accuracy ,"%"

forecast_set    =clf.predict(X_lately)
forecast_set    =np.array(forecast_set)
df['Forecast']  = np.nan

# find the last date to show the original data untill the latest date
last_date   = df_original.iloc[-1].name
last_unix   = time.mktime(last_date.timetuple())
one_day     = 86400
next_unix   = last_unix+one_day

# assign prediction label for the future date
for i in forecast_set:
 next_date = datetime.datetime.fromtimestamp(next_unix)
 next_unix += one_day 
 print "Date:", next_date, ", Price:", i
 df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
 
# plot the results
df_original['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()