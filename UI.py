# load library
import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime
import cufflinks as cf

# print title of web app
st.title("Stock Market Analysis and Prediction")



# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

#values
user_input = st.text_input('Enter Stock Ticker')
date_inputA = st.date_input('Enter Start Date')
date_inputB = st.date_input('Enter End Date')

# Load data from yahoo finance.

data=pdr.get_data_yahoo(user_input, date_inputA, date_inputB)

#fill nan vale with next value within columns
data.fillna(method="ffill",inplace=True)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# create checkbox
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
 
# show the description of data
st.subheader('Detail description about Datasets:-')
descrb=data.describe()
st.write(descrb)

#create new columns like year, month, day
data["Year"]=data.index.year
data["Month"]=data.index.month
data["Weekday"]=data.index.day_name()

# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
st.line_chart(data[["Open","Close"]])

# display plot of Adj Close column in datasets
st.subheader('Graph of Adjacent Close:-')
st.line_chart(data['Adj Close'])

# display plot of volume column in datasets
st.subheader('Graph of Volume:-')
st.line_chart(data['Volume'])

# create new cloumn for data analysis.
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# display the new dataset after modificaton
st.subheader('Newly format DataSet:-')
st.dataframe(data.tail(500))

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['label'])

# split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# display the accuracy of forecast value.
st.subheader('Accuracy:')
st.write(confidence)

forecast_set = clf.predict(X_lately)
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]
    last_date = data.iloc[-1].name
    dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')
    index = 1
for i in forecast_set:
    data.loc[dti[index]] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    index +=1

# display the forecast value.
st.subheader('Forecast value :-')
st.dataframe(data.tail(50))

# display the graph of adj close and forecast columns
st.subheader('Graph of Adj Close and Forecast :-')
st.line_chart(data[["Adj Close","Forecast"]])


# Fetch the data for specified ticker e.g. AAPL from yahoo finance
df_ticker = pdr.DataReader(user_input, 'yahoo', date_inputA, date_inputB)

st.header(f'{user_input} Stock Price')



# Interactive data visualizations using cufflinks
# Create candlestick chart 
qf = cf.QuantFig(df_ticker, legend='top', name=user_input)


# Technical Analysis Studies can be added on demand
# Add Relative Strength Indicator (RSI) study to QuantFigure.studies
qf.add_rsi(periods=20, color='java')

# Add Bollinger Bands (BOLL) study to QuantFigure.studies
qf.add_bollinger_bands(periods=20,boll_std=2,colors=['magenta','grey'],fill=True)

# Add 'volume' study to QuantFigure.studies
qf.add_volume()

fig = qf.iplot(asFigure=True, dimensions=(800, 600))

# Render plot using plotly_chart
st.plotly_chart(fig)