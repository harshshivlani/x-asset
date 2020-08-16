import requests
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import plotly
import plotly.express as px
import plotly.graph_objects as go
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
from bs4 import BeautifulSoup 
import csv
from yahooquery import Ticker
from plotly.subplots import make_subplots


st.write("""
# Cross Asset Market Analytics
""")

from pandas.tseries import offsets
one_m = date.today() - datetime.timedelta(30)
three_m = date.today() - datetime.timedelta(90)
six_m = date.today() - datetime.timedelta(120)
one_yr = date.today() - datetime.timedelta(370)
ytd = date.today() - offsets.YearBegin()
year = date.today().year
yest = date.today() - datetime.timedelta(1)
now = datetime.datetime.now()
now = now.strftime("%b %d, %Y %H:%M")

tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)

components.iframe("https://harshshivlani.github.io/x-asset/liveticker")

#Define function to fetch historical data from Investing.com
def hist_data(name, country):
    df = investpy.get_etf_historical_data(etf=name, country=country, from_date=oneyr, to_date=tdy)['Close']
    df = pd.DataFrame(df)
    df.columns = [name]
    return df

def hist_data_comd(name):
    df = investpy.get_commodity_historical_data(commodity=name, from_date=oneyr, to_date=tdy)['Close']
    df = pd.DataFrame(df)
    df.columns = [name]
    return df

def hist_data_india(name):
    df = investpy.get_index_historical_data(index=name, country='India', from_date=oneyr, to_date=tdy)['Close']
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
    if asset_class=='Commodities':
        for i in range(len(etf_list)):
                df = df.join(hist_data_comd(etf_list[asset_class][i]), on='Date')

    elif asset_class=='Indian Equities':
        for i in range(len(etf_list)):
                df = df.join(hist_data_india(etf_list[asset_class][i]), on='Date')

    else:
        for i in range(len(etf_list)):
                df = df.join(hist_data(etf_list[asset_class][i], etf_list['Country'][i]), on='Date')

    #Forward fill for any missing days i.e. holidays
    #df = df.iloc[:-1,:].ffill().dropna()
    df = df[:yest].ffill().dropna()
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
    df = pd.DataFrame(index=pd.bdate_range(start=one_yr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    df1 = Ticker(list(etf_list['Ticker']), asynchronous=True).history(start=date(date.today().year -1 , date.today().month, date.today().day))['adjclose']
    df1 = pd.DataFrame(df1).unstack().T.reset_index(0).drop('level_0', axis=1)
    df1.index.name = 'Date'
    df1.index = pd.to_datetime(df1.index)
    df = df.merge(df1, on='Date')
    #Forward fill for any missing days i.e. holidays
    df = df.ffill().dropna()
    df.index.name = 'Date'
    df.columns = list(etf_list[asset_class])
    return df

@st.cache
def yield_curve(country='United States'):    
    df = investpy.bonds.get_bonds_overview(country=country)
    df.set_index('name', inplace=True)
    if country=='United States':
        df.index = df.index.str.strip('U.S.')
    elif country =='United Kingdom':
        df.index = df.index.str.strip('U.K.')
    else:
        df.index = df.index.str.strip(country)
    return df['last']


def show_yc():
    us = yield_curve('United States')
    uk = yield_curve('United Kingdom')
    china = yield_curve('China')
    aus = yield_curve('Australia')
    germany = yield_curve('Germany')
    japan = yield_curve('Japan')
    can = yield_curve('Canada')
    ind = yield_curve('India')
    italy = yield_curve('Italy')
    france = yield_curve('France')
    rus = yield_curve('Russia')
    phil = yield_curve('Philippines')
    thai = yield_curve('Thailand')
    brazil = yield_curve('Brazil')
    
    fig = make_subplots(
        rows=14, cols=1,
        subplot_titles=("United States", "United Kingdom", "China", "Australia", "Germany", "Japan", "Canada", "India", "Italy", "France", "Russia", "Philippines", "Thailand", "Brazil"))

    fig.add_trace(go.Scatter(x=us.index, y=us, mode='lines+markers', name='US', line_shape='spline'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=uk.index, y=uk, mode='lines+markers', name='UK', line_shape='spline'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=china.index, y=china, mode='lines+markers', name='China', line_shape='spline'),
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=aus.index, y=aus, mode='lines+markers', name='Australia', line_shape='spline'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=germany.index, y=germany, mode='lines+markers', name='Germany', line_shape='spline'),
                  row=5, col=1)

    fig.add_trace(go.Scatter(x=japan.index, y=japan, mode='lines+markers', name='Japan', line_shape='spline'),
                  row=6, col=1)

    fig.add_trace(go.Scatter(x=can.index, y=can, mode='lines+markers', name='Canada', line_shape='spline'),
                  row=7, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind, mode='lines+markers', name='India', line_shape='spline'),
                  row=8, col=1)

    fig.add_trace(go.Scatter(x=italy.index, y=italy, mode='lines+markers', name='Italy', line_shape='spline'),
                  row=9, col=1)

    fig.add_trace(go.Scatter(x=france.index, y=france, mode='lines+markers', name='France', line_shape='spline'),
                  row=10, col=1)

    fig.add_trace(go.Scatter(x=brazil.index, y=brazil, mode='lines+markers', name='Brazil', line_shape='spline'),
                  row=11, col=1)

    fig.add_trace(go.Scatter(x=thai.index, y=thai, mode='lines+markers', name='Thailand', line_shape='spline'),
                  row=12, col=1)

    fig.add_trace(go.Scatter(x=phil.index, y=phil, mode='lines+markers', name='Philippines', line_shape='spline'),
                  row=13, col=1)

    fig.add_trace(go.Scatter(x=rus.index, y=rus, mode='lines+markers', name='Russia', line_shape='spline'),
                  row=14, col=1)

    fig.update_layout(height=5000, width=400,
                      title_text="Global Sovereign Yield Curves", showlegend=False)
    fig.update_yaxes(title_text="Yield (%)", showgrid=True, zeroline=True, zerolinecolor='red', tickformat = '.3f')
    fig.update_xaxes(title_text="Maturity (Yrs)")
    fig.update_layout(font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f")
                  ,plot_bgcolor = 'White', hovermode='x')
    fig.update_traces(hovertemplate='Maturity: %{x} <br>Yield: %{y:.3f}%')
    fig.update_yaxes(automargin=True)

    return fig

@st.cache
def ytm(country, maturity):
        df = pd.DataFrame(investpy.get_bond_historical_data(bond= str(country)+' '+str(maturity), from_date=oneyr, to_date=tdy)['Close'])
        df.columns = [str(country)]
        df.index = pd.to_datetime(df.index)
        return pd.DataFrame(df)


@st.cache(allow_output_mutation=True)
def global_yields(countries=['U.S.', 'Germany', 'U.K.', 'Italy', 'France', 'Canada', 'China', 'Australia', 'Japan', 'India', 'Russia', 'Brazil', 'Philippines', 'Thailand']):
    """
    
    """
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    
    tens = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    tens.index.name='Date'

    fives = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    fives.index.name='Date'

    twos = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    twos.index.name='Date'

    cntry = countries
    
    for i in range(len(cntry)):
                tens = tens.merge(ytm(cntry[i], '10Y'), on='Date')
    for i in range(len(cntry)):
                fives = fives.merge(ytm(cntry[i], '5Y'), on='Date')
    for i in range(len(cntry)):
                twos = twos.merge(ytm(cntry[i], '2Y'), on='Date')
      
    ytd = date.today() - offsets.YearBegin()
    #10 Year
    teny = pd.DataFrame(data= (tens.iloc[-1,:], tens.diff(1).iloc[-1,:]*100, tens.diff(1).iloc[-5,:]*100, (tens.iloc[-1,:] - tens[ytd:].iloc[0,:])*100, (tens.iloc[-1,:]-tens.iloc[0,:])*100))
    teny = teny.T
    cols = [('10Y', 'Yield'),('10Y', '1 Day'), ('10Y', '1 Week'), ('10Y', 'YTD'), ('10Y', '1 Year')]
    teny.columns = pd.MultiIndex.from_tuples(cols)
    teny.index.name='Countries'
    
    #5 Year
    fivey = pd.DataFrame(data= (fives.iloc[-1,:], fives.diff(1).iloc[-1,:]*100, fives.diff(1).iloc[-6,:]*100,(fives.iloc[-1,:] - fives[ytd:].iloc[0,:])*100, (fives.iloc[-1,:]-fives.iloc[0,:])*100))
    fivey = fivey.T
    cols = [('5Y', 'Yield'),('5Y', '1 Day'), ('5Y', '1 Week'), ('5Y', 'YTD'), ('5Y', '1 Year')]
    fivey.columns = pd.MultiIndex.from_tuples(cols)
    fivey.index.name='Countries'
    
    #2 Year
    twoy = pd.DataFrame(data= (twos.iloc[-1,:], twos.diff(1).iloc[-1,:]*100, twos.diff(1).iloc[-6,:]*100, (twos.iloc[-1,:] - twos[ytd:].iloc[0,:])*100, (twos.iloc[-1,:]-twos.iloc[0,:])*100))
    twoy = twoy.T
    cols = [('2Y', 'Yield'),('2Y', '1 Day'), ('2Y', '1 Week'), ('2Y', 'YTD'), ('2Y', '1 Year')]
    twoy.columns = pd.MultiIndex.from_tuples(cols)
    twoy.index.name='Countries'
    
    yields = twoy.merge(fivey, on='Countries').merge(teny, on='Countries')
    
    data = yields.style.format('{0:,.3f}%', subset=[('2Y', 'Yield'), ('5Y', 'Yield'), ('10Y', 'Yield')])\
            .background_gradient(cmap='RdYlGn_r', subset=list(yields.columns.drop(('2Y', 'Yield')).drop(('5Y', 'Yield')).drop(('10Y', 'Yield')))).set_precision(2)
    return data



#SORTED AND CONDITIONALLY FORMATTED RETURNS DATAFRAME
def returns_hmap(data, cat, asset_class, start=date(2020,3,23), end=date.today(), sortby='1-Day'):
    """
    data = Price Data for the ETFs (dataframe)
    asset_class = asset class of the ETF (str)
    cat = subset or category (list), default is all ETFs mentioned
    """
    st.subheader("Multi Timeframe Returns of " + str(asset_class) + " ETFs")
    st.markdown("Data as of :  " + str(data.index[-1].strftime("%b %d, %Y")))
    df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:],
                              data.pct_change(63).iloc[-1,:], data[str(year):].iloc[-1,:]/data[str(year):].iloc[0,:]-1, data[start:end].iloc[-1,:]/data[start:end].iloc[0,:]-1,
                              data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], drawdowns(data)))
    df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'Custom', '6-Month', '1-Year', 'Max DD']
    df_perf = (df.T*100)
    df_perf.index.name = asset_class

    #Add Ticker Names and sort the dataframe
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)
    if asset_class=='Indian Equities':
        df2 = df_perf.copy()
        df2  = df2.sort_values(by=sortby, ascending=False)
        df2 = df2.round(2).style.format('{0:,.2f}%')\
                     .background_gradient(cmap='RdYlGn')\
                     .set_properties(**{'font-size': '10pt',})
    else:
        if st.checkbox("Show Tickers"):
            tickers = pd.DataFrame(etf_list['Ticker'])
            tickers.index = etf_list[asset_class]
            df2 = tickers.merge(df_perf, on=asset_class)
            df2  = df2.sort_values(by=sortby, ascending=False)
            df2 = df2.round(2).style.format('{0:,.2f}%', subset=list(df2.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(df2.drop(['Ticker'], axis=1).columns))\
                     .set_properties(**{'font-size': '10pt',})
        else: 
            df2 = df_perf
            df2  = df2.sort_values(by=sortby, ascending=False)
            df2 = df2.round(2).style.format('{0:,.2f}%')\
                     .background_gradient(cmap='RdYlGn')\
                     .set_properties(**{'font-size': '10pt',})
    
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
                      yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=12, color="#7f7f7f"),
                      legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '%', width=600, height=600,
                      legend=dict(
                                   orientation="h",
                                    yanchor="bottom",
                                    y=-1,
                                    xanchor="right",
                                    x=1
                                ), margin=dict(l=0,r=0,b=0,t=20,pad=1))
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
            colorscale='rdylgn', hovertemplate='Date: %{x}<br>Security: %{y}<br>Return Z-Score: %{z}<extra></extra>', colorbar = dict(title='Return Z-Score')))

    fig.update_layout(xaxis_nticks=20, font=dict(family="Segoe UI, monospace", size=12, color="#7f7f7f"),
                        margin=dict(l=0,r=0,b=0,t=30,pad=1), width=700, height=500)
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
reits = import_data_yahoo('REIT')
worldeq = import_data_yahoo('World Equities')
fixedinc = import_data_yahoo('Fixed Income')
sectoral = import_data('Sectoral')
fx = import_data_yahoo('Currencies')
comd = import_data('Commodities')
indiaeq = import_data('Indian Equities')

@st.cache(allow_output_mutation=True)
def world_indices():
    world_indices = etf.updated_world_indices('Major')
    return world_indices

world_indices1 = world_indices()

from scipy import stats

def world_id_plots(wdx):
    df = ((world_indices1[0][wdx]*100).dropna().sort_values(ascending=False))
    daily_usd = (df - df.mean())/df.std()
    fig = px.bar(daily_usd, y=daily_usd, color=daily_usd, color_continuous_scale='rdylgn', orientation='v')
    fig.update_layout(xaxis_title='Indices',
                           yaxis_title='Cross Sectional Z-Score', font=dict(family="Segoe UI, monospace", size=11, color="#7f7f7f"),
                           legend_title_text='Return(%)', plot_bgcolor = 'White', yaxis_tickformat = '{:.2f}%', width=600, height=450)
    fig.update_traces(hovertemplate='Index: %{x} <br>Return: %{y:.2f}')
    fig.update_yaxes(automargin=True, showspikes=True)
    fig.update_xaxes(showspikes=True, tickmode='linear')
    return fig


def display_items(data, asset_class, cat):
    dtype1 = st.selectbox('Data Type: ', ('Multi Timeframe Returns Table', 'Performance Chart', 'Rolling Returns Trend Heatmap', 'All'))
    if dtype1=='Multi Timeframe Returns Table':
        #print(st.write("As of "+ str(data.index[-1])))
        start= st.date_input("Custom Start Date: ", date(2020,3,23))
        end = st.date_input("Custom End Date: ", date.today())
        st.dataframe(returns_hmap(data=data[cat], asset_class=asset_class, cat=cat, start=start, end=end), height=1500)
    elif dtype1=='Performance Chart':
        st.subheader("Price Return Performance")
        start_date = st.selectbox('Select Period', list(disp_opts.keys()), index=3, format_func = format_func, key='chart')
        print(st.plotly_chart(plot_chart(data=data[cat], start_date=start_date, cat=cat)))
    elif dtype1=='Rolling Returns Trend Heatmap':
        st.subheader("Rolling Return Trend Heatmap")
        start_date = st.selectbox('Select Period: ', list(disp_opts.keys()), index=3, format_func = format_func, key='trend')
        inv_opt = st.selectbox('Select Timescale: ', list(inv.keys()), index=0, format_func = format_inv)
        ma = st.number_input('Select Rolling Return Period: ', value=15, min_value=1)
        print(st.plotly_chart(trend_analysis(data=data[cat], cat=cat, start_date=start_date, inv=inv_opt, ma=ma)))
    elif dtype1=='All':
        st.dataframe(returns_hmap(data=data[cat], asset_class=asset_class, cat=cat), height=1500)
        st.subheader("Price Return Performance")
        start_date = st.selectbox('Select Period', list(disp_opts.keys()), index=3, format_func = format_func, key='chart')
        print(st.plotly_chart(plot_chart(data=data[cat], start_date=start_date, cat=cat)))
        st.subheader("Rolling Return Trend Heatmap")
        start_date = st.selectbox('Select Period: ', list(disp_opts.keys()), index=3, format_func = format_func, key='trend')
        inv_opt = st.selectbox('Select Timescale: ', list(inv.keys()), index=0, format_func = format_inv)
        ma = st.number_input('Select Rolling Return Period: ', value=15, min_value=1)
        print(st.plotly_chart(trend_analysis(data=data[cat], cat=cat, start_date=start_date, inv=inv_opt, ma=ma)))



## MACRO DATA ANALYTICS 
        


    
# Display the functions/analytics
st.sidebar.header('User Input Parameters')
side_options = st.sidebar.radio('Analytics App Contents', ('Cross Asset Data', 'Live Cross Asset Summary Data', 'ETF Details', 'Economic Calendar', 'Macroeconomic Data', 'Country Macroeconomic Profile'))

if side_options == 'ETF Details':
    def etf_details():
        ticker_name = st.text_input('Enter Ticker Name', value='URTH')
        asset =  st.selectbox('ETF Asset Class:', ('Equity/REIT ETF', 'Fixed Income ETF'))
        if asset=='Equity/REIT ETF':
            details = st.selectbox('Select Data Type:', ('General Overview', 'Top 15 Holdings', 'Sector Exposure',
                                    'Market Cap Exposure', 'Country Exposure', 'Asset Allocation'))
        elif asset=='Fixed Income ETF':
            details = st.selectbox('Select Data Type:', ('General Overview', 'Top 15 Holdings', 'Bond Sector Exposure',
                                    'Coupon Breakdown', 'Credit Quality Exposure', 'Maturity Profile'))
        return [ticker_name, details, asset]

    etf_details = etf_details()
    st.write(etf.etf_details(etf_details[0].upper(), etf_details[1], etf_details[2]))

elif side_options =='Cross Asset Data':
    def user_input_features():
        asset_class = st.sidebar.selectbox("Select Asset Class:", ("World Indices", "World Equities", "Sectoral Equities", "Indian Equities", "Fixed Income","Global Yield Curves", "REITs", "Currencies", "Commodities"))
        return asset_class

    asset_class = user_input_features()
    st.header(asset_class)
    if asset_class=="Fixed Income":
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/fixedinc-chart", height=500)
        option = st.selectbox('Category: ', ('All Fixed Income', 'Sovereign Fixed Income', 'Corporate Credit', 'High Yield', 'Municipals'))
        st.write('**Note:** All returns are in USD')
        print(display_items(fixedinc, 'Fixed Income', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Securities'])))

    elif asset_class=='World Equities':
        if st.checkbox('Show Live Data'):
            components.iframe("https://harshshivlani.github.io/x-asset/indices", width=670, height=500)
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        option = st.selectbox('Category: ', ('G10', 'Emerging Markets', 'Asia', 'Europe', 'Commodity Linked', 'All Countries'))
        st.write('**Note:** All returns are in USD')
        print(display_items(worldeq, 'World Equities', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Countries'])))

    elif asset_class=='Sectoral Equities':
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        option = st.selectbox('Category: ', ('United States', 'Eurozone', 'China', 'Canada', 'Australia'))
        st.write('**Note:** Sectoral Equity ETF Returns are in local currency, except China and US ETFs which are in USD')
        print(display_items(sectoral, 'Sectoral',  cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Sectors'])))

    elif asset_class=='Indian Equities':
        option = st.selectbox('Category: ', ('All Indian Indices', 'Indian Sectoral', 'Indian Strategy Indices'))
        st.write('**Note:** All returns are in INR')
        print(display_items(indiaeq, 'Indian Equities', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Securities'])))

    elif asset_class=='Global Yield Curves':
        opt = st.selectbox('Data Type: ', ('Global Yields Table', 'Yield Curve Charts'))
        if opt=='Yield Curve Charts':
            st.write('As of '+str(date.today().strftime("%b %d, %Y")))
            st.plotly_chart(show_yc())
        else:
            st.write('Global Sovereign Yields: 2 Year, 5 Year and 10 Year Maturity')
            st.write('Note: Yields are denoted in % terms. Change is denoted in basis points (bps)')
            st.write('As of '+ str(date.today().strftime("%b %d, %Y")))
            st.dataframe(global_yields(), width=1500, height=1000)

    elif asset_class=='REITs':
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        st.write('**Note:** All returns are in USD')
        print(display_items(reits, 'REIT', cat=list(reits.columns)))

    elif asset_class=='Currencies':
        if st.checkbox('Show Live Data'):
            components.iframe("https://harshshivlani.github.io/x-asset/cur", width=670, height=500)
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        print(display_items(fx, 'Currencies', cat=list(fx.columns)))

    elif asset_class=='Commodities':
        if st.checkbox('Show Live Data'):
            components.iframe("https://harshshivlani.github.io/x-asset/comd", width=670, height=500)
        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        st.write('**Note:** All returns are in USD')
        print(display_items(comd, 'Commodities', cat=list(comd.columns)))

    elif asset_class=='World Indices':
        if st.checkbox('Show World Indices Map'):
            st.subheader('World Equity Market USD Returns Heatmap (EOD)')
            ret_type = st.selectbox('Return Period: ', ('1-Day', '1-Week', '1-Month', 'YTD'))
            iso = pd.read_excel('World_Indices_List.xlsx', sheet_name='iso')
            iso.set_index('Country', inplace=True)
            data2 = etf.format_world_data(world_indices1[0], country='Yes')[0].merge(iso['iso_alpha'], on='Country')
            data2[['1-Day', '1-Week', '1-Month', 'YTD']] = data2[['1-Day', '1-Week', '1-Month', 'YTD']].round(4)*100

            df = data2
            for col in df.columns:
                df[col] = df[col].astype(str)
            
            df['text'] = 'Return: '+df[ret_type]+'%' + '<br>' \
                          'Country: '+ df['Country'] + '<br>' \
                           #'Index: '+ data2['Indices'].astype('str')
            fig1 = px.choropleth(df, locations="iso_alpha",
                                color=ret_type,
                                hover_name="Country",
                                color_continuous_scale='RdYlGn')
            fig1 = go.Figure(data=go.Choropleth(locations=df['iso_alpha'], z=df[ret_type].astype(float).round(2), colorscale='RdYlGn', autocolorscale=False,
                text=df['text']))



            fig1.update_layout(width=420, height=300, margin=dict(l=0,r=0,b=0,t=0,pad=1),
                                xaxis=dict(scaleanchor='x', constrain='domain'),
                  coloraxis_colorbar_x=1)
            st.plotly_chart(fig1)

        if st.checkbox('Show Live Markets'):
            components.iframe("https://harshshivlani.github.io/x-asset/indices", width=670, height=500)

        if st.checkbox('Show Live Chart'):
            components.iframe("https://harshshivlani.github.io/x-asset/equity-chart", height=500)
        
        st.subheader("Multi-TimeFrame Return Table:")
        usd = st.selectbox('Currency: ', ('USD', 'Local Currency'))
        st.write('As of ' + str(world_indices1[1].strftime("%b %d, %Y")))
        if st.checkbox("Show Countries"):
            print(st.dataframe(etf.format_world_data(world_indices1[0], usd=usd, country='Yes')[1], height=1000))
        else:
            print(st.dataframe(etf.format_world_data(world_indices1[0], usd=usd, country='No')[1], height=1000))
        if st.checkbox("Show Returns Z-Score Bar Plot"):
            wdx = st.selectbox('Plot Data Type: ', ('$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)'))
            fig = world_id_plots(wdx)
            fig.update_layout(margin=dict(l=0,r=0,b=0,t=0,pad=1))
            #fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig)

elif side_options=='Macroeconomic Data':
     st.subheader('Macroeconomic Data')
     cat = st.selectbox('Select Data Category: ', ('World Manufacturing PMIs', 'GDP', 'Retail Sales', 'Inflation', 'Unemployment'))
     if cat == 'World Manufacturing PMIs':
         st.subheader('World Manufacturing PMIs')
         continent = st.selectbox('Select Continent', ('World', 'G20', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.world_pmis(continent=continent), width=1000, height=1500)
     elif cat == 'GDP':
         st.subheader('World GDP Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.gdp(continent=continent), width=1200, height=2000)
     elif cat=='Retail Sales':
         st.subheader('Retail Sales')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         time = st.selectbox('Select Period: ', ('YoY', 'MoM'))
         st.dataframe(etf.retail(continent=continent, time=time), width=1200, height=2000)
     elif cat == 'Inflation':
         st.subheader('World Inflation Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.inflation(continent=continent), width=1200, height=2000)
     elif cat == 'Unemployment':
         st.subheader('World Unemployment Data')
         continent = st.selectbox('Select Continent', ('G20', 'World', 'America', 'Europe', 'Asia', 'Africa'))
         st.dataframe(etf.unemp(continent=continent), width=1200, height=2000)

elif side_options == 'Economic Calendar':
     st.subheader('Economic Calendar')
     components.iframe("https://harshshivlani.github.io/x-asset/ecocalendar", height=800)
     #importances = st.multiselect('Importance: ', ['Low', 'Medium', 'High'], ['Medium', 'High'])
     #st.dataframe(etf.eco_calendar(importances=importances), width=2000, height=1200)

elif side_options == 'Country Macroeconomic Profile':
     st.subheader('Country Macroeconomic Profile')
     countries_list = st.selectbox('Select Country: ', ["United-States", "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua-and-Barbuda","Argentina","Armenia","Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bermuda","Bhutan","Bolivia","Bosnia-and-Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina-Faso","Burundi","Cambodia","Cameroon","Canada","Cape-Verde","Cayman-Islands","Central-African-Republic","Chad","Chile","China","Colombia","Comoros","Congo","Costa-Rica","Croatia","Cuba","Cyprus","Czech-Republic","Denmark","Djibouti","Dominica","Dominican-Republic","East-Timor","Ecuador","Egypt","El-Salvador","Equatorial-Guinea","Eritrea","Estonia","Ethiopia","Euro-Area","Faroe-Islands","Finland","France","Gabon","Gambia","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","Hong-Kong","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Isle-of-Man","Israel","Italy","Ivory-Coast","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kosovo","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Macao","Macedonia","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Mauritania","Mauritius","Mexico","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands","New-Zealand","Nicaragua","Niger","Nigeria","North-Korea","Norway","Oman","Pakistan","Palestine","Panama","Paraguay","Peru","Philippines","Poland","Portugal","Puerto-Rico","Qatar","Republic-of-the-Congo","Romania","Russia","Rwanda","Sao-Tome-and-Principe","Saudi-Arabia","Senegal","Serbia","Seychelles","Sierra-Leone","Singapore","Slovakia","Slovenia","Somalia","South-Africa","South-Korea","South-Sudan","Spain","Sri-Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Trinidad-and-Tobago","Tunisia","Turkey","Turkmenistan","Uganda","Ukraine","United-Arab-Emirates","United-Kingdom","United-States","Uruguay","Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"])
     data_type = st.selectbox('Data Category: ', ['Overview', 'GDP', 'Labour', 'Inflation', 'Money', 'Trade', 'Government', 'Taxes', 'Business', 'Consumer'])
     st.dataframe(etf.country_macros(country=countries_list, data_type=data_type), height=1200)

elif side_options == 'Live Cross Asset Summary Data':
    st.subheader('Updated Cross Asset Market Performance')
    cat = st.selectbox('Asset Class: ', ('World Indices', 'Industry Stocks: India', 'Commodities', 'Currencies'))
    st.write('As of ' + str(now))
    if cat=='World Indices':
        st.dataframe(etf.live_indices(), height=1000)

    elif cat=='Industry Stocks: India':

        inv = {'energy':'Energy', 'financial': 'Financial', 'healthcare':'Healthcare', 'technology': 'IT','telecom_utilities': 'Telecom & Utilities',
         'fmcg':'FMCG', 'realty': 'Realty', 'manufacturing_materials': 'Basic Materials', 'consumer_durables':'Consumer Durables',
          'industrials': 'Industrials', 'power':'Power', 'auto':'Auto'}
        def format_inv(option):
            return inv[option]

        inds = st.selectbox('Industry: ',list(inv.keys()), index=0, format_func = format_inv)
        st.dataframe(etf.india_inds(industry=inds), height=1000, width=1000)

    elif cat=='Commodities':
        st.dataframe(etf.live_comds(), height=1300, width=1300)

    elif cat=='Currencies':
        st.dataframe(etf.live_curr(), height=1300, width=1300)        


st.sidebar.markdown('Developed by Harsh Shivlani')