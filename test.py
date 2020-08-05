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

tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)

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
    df = df.iloc[:-1,:].ffill().dropna()
    #df = df[:yest].ffill().dropna()
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
    st.markdown("Data as of :  " + str(data.index[-1].strftime("%b %d, %Y")))
    df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:],
                              data.pct_change(63).iloc[-1,:], data[str(year):].iloc[-1,:]/data[str(year):].iloc[0,:]-1, data[str(year):].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1,
                              data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], drawdowns(data)))
    df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'From 23rd March', '6-Month', '1-Year', 'Drawdowns']
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
        tickers = pd.DataFrame(etf_list['Ticker'])
        tickers.index = etf_list[asset_class]
        df2 = tickers.merge(df_perf, on=asset_class)
        df2  = df2.sort_values(by=sortby, ascending=False)
        df2 = df2.round(2).style.format('{0:,.2f}%', subset=list(df2.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(df2.drop(['Ticker'], axis=1).columns))\
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
                      yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                      legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '%', width=1300, height=650)
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
comd = import_data('Commodities')
indiaeq = import_data('Indian Equities')

@st.cache(allow_output_mutation=True)
def world_indices():
    world_indices = etf.updated_world_indices('Major')
    return world_indices



def world_id_plots(wdx):
    daily_usd = ((world_indices()[wdx]*100).dropna().sort_values(ascending=False))
    fig = px.bar(daily_usd, color=daily_usd, color_continuous_scale='rdylgn', text=world_indices().sort_values(by=wdx, ascending=False)['Country'])
    fig.update_layout(title = 'World Indices Performance (%) (EOD)',
                           xaxis_title='Indices',
                           yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                           legend_title_text='Return(%)', plot_bgcolor = 'White', yaxis_tickformat = '{:.2f}%', width=1300, height=650, hovermode='x')
    fig.update_traces(hovertemplate='Index: %{x} <br>Return: %{y:.2f}%')
    fig.update_yaxes(automargin=True)
    return fig


def display_items(data, asset_class, cat):
    dtype1 = st.selectbox('Data Type: ', ('Multi Timeframe Returns Table', 'Performance Chart', 'Rolling Returns Trend Heatmap', 'All'))
    if dtype1=='Multi Timeframe Returns Table':
        #print(st.write("As of "+ str(data.index[-1])))
        st.dataframe(returns_hmap(data=data[cat], asset_class=asset_class, cat=cat), height=1500)
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
side_options = st.sidebar.radio('Analytics App Contents', ('Cross Asset Data', 'ETF Details', 'Economic Calendar', 'Macroeconomic Data', 'Country Macroeconomic Profile'))

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
        asset_class = st.sidebar.selectbox("Select Asset Class:", ("World Indices", "World Equities", "Sectoral Equities", "Indian Equities", "Fixed Income", "REITs", "Currencies", "Commodities"))
        return asset_class

    asset_class = user_input_features()
    st.header(asset_class)
    if asset_class=="Fixed Income":
        option = st.selectbox('Category: ', ('All Fixed Income', 'Sovereign Fixed Income', 'Corporate Credit', 'High Yield', 'Municipals'))
        st.write('**Note:** All returns are in USD')
        print(display_items(fixedinc, 'Fixed Income', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Securities'])))

    elif asset_class=='World Equities':
        option = st.selectbox('Category: ', ('All Countries', 'Emerging Markets', 'Asia', 'G10', 'Europe', 'Commodity Linked'))
        st.write('**Note:** All returns are in USD')
        print(display_items(worldeq, 'World Equities', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Countries'])))

    elif asset_class=='Sectoral Equities':
        option = st.selectbox('Category: ', ('United States', 'Eurozone', 'China', 'Canada', 'Australia'))
        st.write('**Note:** Sectoral Equity ETF Returns are in local currency, except China and US ETFs which are in USD')
        print(display_items(sectoral, 'Sectoral',  cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Sectors'])))

    elif asset_class=='Indian Equities':
        option = st.selectbox('Category: ', ('All Indian Indices', 'Indian Sectoral', 'Indian Strategy Indices'))
        st.write('**Note:** All returns are in INR')
        print(display_items(indiaeq, 'Indian Equities', cat=list(pd.read_excel('etf_names.xlsx', sheet_name=option)['Securities'])))

    elif asset_class=='REITs':
        st.write('**Note:** All returns are in USD')
        print(display_items(reits, 'REIT', cat=list(reits.columns)))

    elif asset_class=='Currencies':
        print(display_items(fx, 'Currencies', cat=list(fx.columns)))

    elif asset_class=='Commodities':
        st.write('**Note:** All returns are in USD')
        print(display_items(comd, 'Commodities', cat=list(comd.columns)))

    elif asset_class=='World Indices':
        if st.checkbox('Show World Indices Map'):
            st.subheader('World Equity Market USD Returns Heatmap (EOD)')
            ret_type = st.selectbox('Return Period: ', ('$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)'))
            iso = pd.read_excel('World_Indices_List.xlsx', sheet_name='iso')
            iso.set_index('Country', inplace=True)
            data2 = etf.format_world_data(world_indices())[0].merge(iso['iso_alpha'], on='Country')
            data2[['$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)']] = data2[['$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)']].round(4)*100

            df = data2
            fig1 = px.choropleth(df, locations="iso_alpha",
                                color=ret_type,
                                hover_name="Country",
                                color_continuous_scale='RdYlGn')
            fig1.update_layout(width=1000, height=650)
            st.plotly_chart(fig1)

        usd = st.selectbox('Currency: ', ('USD', 'Local Currency'))
        print(st.dataframe(etf.format_world_data(world_indices(), usd=usd)[1]))
        wdx = st.selectbox('Plot Data Type: ', ('$ 1D Chg (%)', '$ 1W Chg (%)', '$ 1M Chg (%)', '$ Chg YTD (%)', '1D Chg (%)', '1W Chg (%)', '1M Chg (%)', 'Chg YTD (%)'))
        st.plotly_chart(world_id_plots(wdx), width=2000, height=1500)

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
     importances = st.multiselect('Importance: ', ['Low', 'Medium', 'High'], ['Medium', 'High'])
     st.dataframe(etf.eco_calendar(importances=importances), width=2000, height=1200)

elif side_options == 'Country Macroeconomic Profile':
     st.subheader('Country Macroeconomic Profile')
     countries_list = st.selectbox('Select Country: ', ["United-States", "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua-and-Barbuda","Argentina","Armenia","Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bermuda","Bhutan","Bolivia","Bosnia-and-Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina-Faso","Burundi","Cambodia","Cameroon","Canada","Cape-Verde","Cayman-Islands","Central-African-Republic","Chad","Chile","China","Colombia","Comoros","Congo","Costa-Rica","Croatia","Cuba","Cyprus","Czech-Republic","Denmark","Djibouti","Dominica","Dominican-Republic","East-Timor","Ecuador","Egypt","El-Salvador","Equatorial-Guinea","Eritrea","Estonia","Ethiopia","Euro-Area","Faroe-Islands","Finland","France","Gabon","Gambia","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","Hong-Kong","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Isle-of-Man","Israel","Italy","Ivory-Coast","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kosovo","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Macao","Macedonia","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Mauritania","Mauritius","Mexico","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands","New-Zealand","Nicaragua","Niger","Nigeria","North-Korea","Norway","Oman","Pakistan","Palestine","Panama","Paraguay","Peru","Philippines","Poland","Portugal","Puerto-Rico","Qatar","Republic-of-the-Congo","Romania","Russia","Rwanda","Sao-Tome-and-Principe","Saudi-Arabia","Senegal","Serbia","Seychelles","Sierra-Leone","Singapore","Slovakia","Slovenia","Somalia","South-Africa","South-Korea","South-Sudan","Spain","Sri-Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Trinidad-and-Tobago","Tunisia","Turkey","Turkmenistan","Uganda","Ukraine","United-Arab-Emirates","United-Kingdom","United-States","Uruguay","Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"])
     data_type = st.selectbox('Data Category: ', ['Overview', 'GDP', 'Labour', 'Inflation', 'Money', 'Trade', 'Government', 'Taxes', 'Business', 'Consumer'])
     st.dataframe(etf.country_macros(country=countries_list, data_type=data_type), height=1200)


st.sidebar.markdown('Developed by Harsh Shivlani')
