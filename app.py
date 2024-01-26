import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
from pmdarima import auto_arima

from PIL import Image
import plotly.graph_objs as go

image = Image.open('stock.jpeg')

st.image(image, use_column_width=True)

st.markdown('''
# Fintech Stock Price App 
This app shows the closing financial stock price values for various companies along with the timeline.  
- App built by Anshuman Shukla and Pranav Sawant of Team Skillocity.
- Dataset resource: Yahoo Finance
- Added feature: Time series forecasting with ARIMA that can predict the stock price values over 15 years.
- Note: User inputs for the company to be analyzed are taken from the sidebar. It is located at the top left of the page (arrow symbol). Inputs for other features of data analysis can also be provided from the sidebar itself. 
''')
st.write('---')

def load_data():
    # Get a list of S&P 500 companies from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    components = pd.read_html(url, header=0)[0]
    return components

def load_quotes(asset):
    return yf.download(asset, start="2020-01-01", end=date.today())

def main():
    components = load_data()
    
    st.sidebar.title("Options")

    if st.sidebar.checkbox("View companies list"):
        st.dataframe(components)
        
    title = st.empty()
        
    def label(row):
        return f"{row['Symbol']} - {row['Security']}"

    st.sidebar.subheader("Select company")
    asset = st.sidebar.selectbox(
        "Click below to select a new company",
        components.apply(label, axis=1),
        index=3,
    )

    title.title(asset)
    
    # Extract the symbol from the selected row
    selected_symbol = components[components.apply(label, axis=1) == asset]['Symbol'].values[0]
    
    data0 = load_quotes(selected_symbol)
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

    st.subheader("Stock Chart")
    st.line_chart(data2)

    st.subheader("Company Statistics")
    st.table(data2.describe())

    if st.sidebar.checkbox("View Historical Company Shares"):
        st.subheader(f"{selected_symbol} historical data")
        st.write(data2)

main()

#part2
def pre_dict():
    st.header('Stock prediction')

    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = yf.Tickers(['AAPL', 'GOOGL', 'MSFT', 'GME'])
    selected_stock = st.selectbox('Select company for prediction', stocks.tickers)

    n_years = st.slider('Years of prediction:', 1, 15)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)
	
    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def runpls():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
	
    runpls()

    # Predict forecast with ARIMA.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = auto_arima(df_train['y'], seasonal=False, trace=True)
    forecast, conf_int = model.predict(n_periods=period, return_conf_int=True, alpha=0.05)

    # Show and plot forecast
    st.subheader('Forecast data')
    forecast_index = pd.date_range(df_train['ds'].max(), periods=period + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast})
    st.write(forecast_df)

    st.write(f'Forecast plot for {n_years} years')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name='Actual'))
    fig1.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast'))
    fig1.layout.update(title_text='Time Series Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

if st.button('Stock Prediction'): 
   if st.button('Stop Prediction'):
      st.title("Prediction Stopped")
   else:
       pre_dict()

st.sidebar.subheader("Read an article about this app: https://proskillocity.blogspot.com/2021/05/financial-stock-price-web-app.html")
