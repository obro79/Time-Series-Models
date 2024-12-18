import streamlit as st
from homepage import priceChart, logReturnsChart


## TODO add default ticker to prevent null error

st.title("Time Series Forecasting")
st.sidebar.title("SideBar Options")
ticker = st.sidebar.text_input("Stock Ticker")
model_choice = st.sidebar.selectbox("Choose a Model:", ["ARCH", "GARCH", "TGARCH", "EGARCH", "SARIMA", "ARIMA", "ARMA"])

## stock price chart:

st.pyplot(priceChart(ticker))
st.pyplot(logReturnsChart(ticker))


## log returns chart:

## squared returns chart:



# if model_choice == "ARIMA":
    
# elif    


