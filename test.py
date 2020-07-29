import yfinance as yf
import streamlit as st
import ta
import jinja2
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import date
import datetime
import edhec_risk_kit as erk
import yfinance as yf
import etf as etf
import ipywidgets as widgets
import investpy
from ipywidgets import interact, interact_manual
#notebook formatting
from IPython.core.display import display, HTML


st.write("""
# Cross Asset Market Analytics
""")

# Side Bar
st.sidebar.header('User Input Parameters')
def user_input_features():
    asset_class = st.sidebar.selectbox("Select Asset Class:", ("World Equities", "Sectoral Equities", "Fixed Income", "REITs", "Currencies"))
    return asset_class

asset_class = user_input_features()

st.header(asset_class)

from pandas.tseries import offsets
one_m = date.today() - datetime.timedelta(30)
three_m = date.today() - datetime.timedelta(90)
six_m = date.today() - datetime.timedelta(120)
one_yr = date.today() - datetime.timedelta(365)
three_yr = date.today() - datetime.timedelta(365*3)
five_yr = date.today() - datetime.timedelta(365*5)
ytd = date.today() - offsets.YearBegin()
year = date.today().year

tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)

#Define function to fetch historical data from Investing.com
def hist_data(name, country):
    df = investpy.get_etf_historical_data(etf=name, country=country, from_date=oneyr, to_date=tdy)['Close']
    df = pd.DataFrame(df)
    df.columns = [name]
    return df

def drawdowns(data):
    """
    Max Drawdown in the current calendar year
    """
    return_series = pd.DataFrame(data.pct_change().dropna()[str(date.today().year):])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)

#Raw Data Extraction of ETFs  mentioned in the excel sheet
@st.cache
def import_data(asset_class):
    """
    Imports Historical Data for Mentioned ETFs
    asset_class = mention the asset class for ETFs data import (str)
    options available = 'Fixed Income', 'REIT', 'Currencies', 'Commodities', 'World Equities', 'Sector Equities'
    """
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    for i in range(len(etf_list)):
            df = df.join(hist_data(etf_list[asset_class][i], etf_list['Country'][i]), on='Date')

    #Forward fill for any missing days i.e. holidays
    df = df.iloc[:-1,:].ffill().dropna()
    df.index.name = 'Date'
    return df

@st.cache
def import_data_yahoo(asset_class):
    """
    Imports Historical Data for Mentioned ETFs
    asset_class = mention the asset class for ETFs data import (str)
    options available = 'Fixed Income', 'REIT', 'Currencies', 'Commodities', 'World Equities', 'Sectoral'
    """
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)
    etf_list = etf_list.sort_values(by='Ticker')

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    df1 = yf.download(list(etf_list['Ticker']), start=one_yr, progress=False)['Adj Close']
    df = df.merge(df1, on='Date')
    #Forward fill for any missing days i.e. holidays
    df = df.ffill().dropna()
    df.index.name = 'Date'
    df.columns = list(etf_list[asset_class])
    return df

#SORTED AND CONDITIONALLY FORMATTED RETURNS DATAFRAME
def returns_hmap(data, cat, asset_class, sortby='1-Day'):
    """
    data = Price Data for the ETFs (dataframe)
    asset_class = asset class of the ETF (str)
    cat = subset or category (list), default is all ETFs mentioned
    """
    st.subheader("Multi Timeframe Returns of " + str(asset_class) + " ETFs")
    df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:],
                              data.pct_change(63).iloc[-1,:], data[str(year):].iloc[-1,:]/data[str(year):].iloc[0,:]-1, data[str(year):].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1,
                              data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], drawdowns(data)))
    df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'From 23rd March', '6-Month', '1-Year', 'Drawdowns']
    df_perf = (df.T*100)
    df_perf.index.name = asset_class

    #Add Ticker Names and sort the dataframe
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)
    tickers = pd.DataFrame(etf_list['Ticker'])
    tickers.index = etf_list[asset_class]
    df2 = tickers.merge(df_perf, on=asset_class)
    df2  = df2.sort_values(by=sortby, ascending=False)
    return df2

#PLOT RETURN CHART BY PLOTLY
def plot_chart(data, cat, start_date=one_yr):
    """
    Returns a Plotly Interactive Chart for the given timeseries data (price)
    data = price data for the ETFs (dataframe)
    """
    df = ((((1+data[cat].dropna()[start_date:date.today()].pct_change().fillna(0.00))).cumprod()-1)).round(4)
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                      legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%', width=1300, height=650)
    fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
    fig.update_yaxes(automargin=True)
    return fig

#TREND ANALYSIS
def trend_analysis(data, cat, start_date=one_yr, inv='B', ma=15):
    """
    data = price data (dataframe)
    inv = daily to be resampled to weekly ('W') or monthly ('BM') return data
    ma = rolling return lookback period, chose 1 for just daily/monthly etc based incase you resample via inv variable
    cat =  any subset of data (list - column names of data)
    """
    d = (data[cat].pct_change(ma).dropna()[start_date:date.today()].resample(inv).agg(lambda x: (x + 1).prod() - 1).round(4)*100)
    fig = go.Figure(data=go.Heatmap(
            z=((d - d.mean())/d.std()).round(2).T.values,
            x=((d - d.mean())/d.std()).index,
            y=list(data[cat].columns), zmax=3, zmin=-3,
            colorscale='rdylgn', hovertemplate='Date: %{x}<br>ETF: %{y}<br>Return Z-Score: %{z}<extra></extra>', colorbar = dict(title='Return Z-Score')))

    fig.update_layout(xaxis_nticks=20, font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"), width=1300, height=650)
    return fig

# Additional Settings for Interactive Widget Buttons for Charts & Plots
#Select Time Frame Options
disp_opts = {one_m: '1 Month', three_m: '3 Months', six_m:'6 Months', ytd: 'Year-to-Date', one_yr:'1-Year'} #To show text in options but have numbers in the backgroud
def format_func(option):
    return disp_opts[option]

#Select Daily/Weekly/Monthly data for Trend Analysis
inv = {'B': 'Daily', 'W': 'Weekly', 'BM': 'Monthly'}
def format_inv(option):
    return inv[option]


#Import all raw price data ready
reits = import_data('REIT')
worldeq = import_data_yahoo('World Equities')
fixedinc = import_data_yahoo('Fixed Income')
sectoral = import_data('Sectoral')
fx = import_data_yahoo('Currencies')

dtype1 = st.selectbox('Data Type: ', ('Multi Timeframe Returns Table', 'Performance Chart', 'Rolling Returns Trend Heatmap', 'All'))

def display_items(data, asset_class):
    if dtype1=='Multi Timeframe Returns Table':
        #print(st.write("As of "+ str(data.index[-1])))
        st.dataframe(returns_hmap(data=data, asset_class=asset_class, cat=list(data.columns)), width=1400, height=600)
    elif dtype1=='Performance Chart':
        st.subheader("Price Return Performance")
        start_date = st.selectbox('Select Period', list(disp_opts.keys()), index=3, format_func = format_func, key='chart')
        print(st.plotly_chart(plot_chart(data=data, start_date=start_date, cat=list(data.columns))))
    elif dtype1=='Rolling Returns Trend Heatmap':
        st.subheader("Rolling Return Trend Heatmap")
        start_date = st.selectbox('Select Period: ', list(disp_opts.keys()), index=3, format_func = format_func, key='trend')
        inv_opt = st.selectbox('Select Timescale: ', list(inv.keys()), index=0, format_func = format_inv)
        ma = st.number_input('Select Rolling Return Period: ', value=15, min_value=1)
        print(st.plotly_chart(trend_analysis(data=data, cat=list(data.columns), start_date=start_date, inv=inv_opt, ma=ma)))
    elif dtype1=='All':
        st.dataframe(returns_hmap(data=data, asset_class=asset_class, cat=list(data.columns)))
        st.subheader("Price Return Performance")
        start_date = st.selectbox('Select Period', list(disp_opts.keys()), index=3, format_func = format_func, key='chart')
        print(st.plotly_chart(plot_chart(data=data, start_date=start_date, cat=list(data.columns))))
        st.subheader("Rolling Return Trend Heatmap")
        start_date = st.selectbox('Select Period: ', list(disp_opts.keys()), index=3, format_func = format_func, key='trend')
        inv_opt = st.selectbox('Select Timescale: ', list(inv.keys()), index=0, format_func = format_inv)
        ma = st.number_input('Select Rolling Return Period: ', value=15, min_value=1)
        print(st.plotly_chart(trend_analysis(data=data, cat=list(data.columns), start_date=start_date, inv=inv_opt, ma=ma)))

# Display the functions/analytics
if asset_class=="Fixed Income":
    print(display_items(fixedinc, 'Fixed Income'))

elif asset_class=='World Equities':
    print(display_items(worldeq, 'World Equities'))

elif asset_class=='Sectoral Equities':
    print(display_items(sectoral, 'Sectoral'))

elif asset_class=='REITs':
    print(display_items(reits, 'REIT'))

elif asset_class=='Currencies':
    print(display_items(fx, 'Currencies'))

else:
    "Please select a valid asset class from the list in the sidebar."
