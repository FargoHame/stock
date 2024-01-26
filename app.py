import streamlit as st
from datetime import date

import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from plotly import graph_objs as go

from PIL import Image
import pandas as pd

import hashlib
import types


image = Image.open('stock.jpeg')

st.image(image, use_column_width=True)

st.markdown('''
# Fintech Stock Price App 
This app shows the closing financial stock price values for S and P 500 companies along with the timeline.  
- These are 500 of the largest companies listed on stock exchanges in the US.
- App built by Anshuman Shukla and Pranav Sawant of Team Skillocity.
- Dataset resource: Yahoo Finance
- Added feature: Time series forecasting with fbprophet that can predict the stock price values over 15 years.
- Note: User inputs for the company to be analysed are taken from the sidebar. It is located at the top left of the page (arrow symbol). Inputs for other features of data analysis can also be provided from the sidebar itself. 
''')
st.write('---')




def load_data():
    components = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    components = components.drop("SEC filings", axis=1) if "SEC filings" in components.columns else components
    return components.set_index("Symbol")


def load_quotes(asset):
    # Use yfinance to fetch historical stock price data
    data = yf.download(asset, start="2010-01-01", end=date.today())
    return data


def main():
    components = load_data()
    
    st.sidebar.title("Options")

    if st.sidebar.checkbox("View companies list"):
        st.dataframe(
            components[["Security", "GICS Sector", "Date first added", "Founded"]]
        )
        
    title = st.empty()
        
    def label(symbol):
        a = components.loc[symbol]
        return symbol + " - " + a.Security

    st.sidebar.subheader("Select company")
    asset = st.sidebar.selectbox(
        "Click below to select a new company",
    components.index.sort_values(),
    index=3,
    format_func=label,
    )

    
    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox("View company info", True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None

    section = st.sidebar.slider(
        "Number of days for Data Analysis of stocks",
        min_value=30,
        max_value=min([5000, data.shape[0]]),
        value=1000,
        step=10,
    )

    data2 = data[-section:]["Adj Close"].to_frame("Adj Close")

    sma = st.sidebar.checkbox("Simple Moving Average")
    if sma:
        period = st.sidebar.slider(
            "Simple Moving Average period", min_value=5, max_value=500, value=20, step=1
        )
        data[f"SMA {period}"] = data["Adj Close"].rolling(period).mean()
        data2[f"SMA {period}"] = data[f"SMA {period}"].reindex(data2.index)

    sma2 = st.sidebar.checkbox("Simple Moving Average 2")
    if sma2:
        period2 = st.sidebar.slider(
            "Simple Moving Average 2 period", min_value=5, max_value=500, value=100, step=1
        )
        data[f"SMA2 {period2}"] = data["Adj Close"].rolling(period2).mean()
        data2[f"SMA2 {period2}"] = data[f"SMA2 {period2}"].reindex(data2.index)

    st.subheader("Stock Chart")
    st.line_chart(data2)

    
    st.subheader("Company Statistics")
    st.table(data2.describe())

    if st.sidebar.checkbox("View Historical Company Shares"):
        st.subheader(f"{asset} historical data")
        st.write(data2)

    
    


main()

#part2
# ... (previous code)


def pre_dict():
    st.header('Stock prediction')

    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ('AAPL', 'GOOGL', 'MSFT', 'GME')  # Updated the stock symbols
    selected_stock = st.selectbox('Select company for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 15)
    period = n_years * 365

    # Load historical stock price data using yfinance
    data = yf.download(selected_stock, start="2010-01-01", end=date.today())
    
    st.subheader('Raw data')
    st.write(data.tail())

    def runpls():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    runpls()

    # Prepare data for SARIMAX model
    df_train = data[['Close']]
    df_train = df_train.rename(columns={"Close": "y"})
    df_train = df_train.asfreq('D')
    df_train = df_train.interpolate()

    # Fit SARIMAX model
    model = SARIMAX(df_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.predict(start=len(df_train), end=len(df_train) + period)

    # Plot forecast
    st.subheader('Forecast data')
    st.write(forecast)

    st.write(f'Forecast plot for {n_years} years')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], name="actual"))
    fig1.add_trace(go.Scatter(x=df_train.index[-30:], y=df_train['y'][-30:], name="last_30_days"))
    fig1.add_trace(go.Scatter(x = (df_train.index[-30:] + pd.to_timedelta(1, unit='D')).astype(np.int64) * period // (365 * 24 * 60 * 60 * 10**9)
,
                              y=forecast, name="forecast"))
    fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    st.write("Not available for SARIMAX model")

pre_dict()
st.sidebar.subheader("Read an article about this app: https://proskillocity.blogspot.com/2021/05/financial-stock-price-web-app.html")
