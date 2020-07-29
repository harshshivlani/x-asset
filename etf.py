#data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#filter warnings for final presentation
import warnings
warnings.filterwarnings("ignore")
import edhec_risk_kit as erk
import yfinance as yf
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from datetime import date
import time
import datetime
import investpy

#notebook formatting
from IPython.core.display import display, HTML

def drawdowns2020(data):
    return_series = pd.DataFrame(data.pct_change().dropna()[str(date.today().year):])
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.min(axis=0)

def returns_heatmap(data, max_drawdowns, title, tickers, sortby='1-Day', reit='No', currencies='No', alok_secs='No', fg_data='No', india='No', style='Yes'):
    """

    """
    if reit=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    elif currencies=='Yes':
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    elif alok_secs=='Yes':
        data = data.ffill()
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020-01-06':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', 'Drawdowns']

    elif fg_data=='Yes':
        now = time.localtime()
        last = datetime.date(now.tm_year, now.tm_mon, 1) - datetime.timedelta(1)
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(7).iloc[-1,:], data.pct_change(30).iloc[-1,:], data.iloc[-1,:]/data[last:].iloc[0,:]-1, data.pct_change(90).iloc[-1,:]))
        df.index = ['1-Day', '1-Week', '1-Month', 'MTD', '3-Month']


    else:
        df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:], data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, data.pct_change(126).iloc[-1,:], data.pct_change(252).iloc[-1,:], data.pct_change(252*3).iloc[-1,:], max_drawdowns))
        df.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', '3-Year', 'Drawdowns']


    df_perf = (df.T*100)
    if india=='No':
        df_perf.insert(loc=0, column='Tickers', value=list(tickers))
        df_perf = df_perf.sort_values(by=sortby, ascending=False)

    #df_perf.insert(loc=1, column='Yields', value=list(yields.iloc[:,0].round(2).values))
    else:
        df_perf = df_perf.sort_values(by=sortby, ascending=False)

    df_perf.index.name = title

    if style=='Yes' and india=='No':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%', subset=list(df_perf.drop(['Tickers'], axis=1).columns))\
                 .background_gradient(cmap='RdYlGn', subset=(df_perf.drop(['Tickers'], axis=1).columns))\
                 .set_properties(**{'font-size': '10pt',})

    elif style=='Yes' and india=='Yes':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%')\
                 .background_gradient(cmap='RdYlGn')\
                 .set_properties(**{'font-size': '10pt',})

    else:
        df_perf = df_perf.round(2)

    return df_perf


def returns_heatmap_alok(data, max_drawdowns, title, sortby='1-Day', style='Yes'):
    """

    """
    data = data.ffill()
    df = pd.DataFrame(data = (data.pct_change(1).iloc[-1,:],  data.pct_change(5).iloc[-1,:], data.pct_change(21).iloc[-1,:], data.pct_change(42).iloc[-1,:], data.pct_change(63).iloc[-1,:], data['2020':].iloc[-1,:]/data['2020-01-06':].iloc[0,:]-1, data['2020':].iloc[-1,:]/data['2020-03-23':].iloc[0,:]-1, max_drawdowns))
    df.index = ['1-Day', '1-Week', '1-Month', '2-Month', '3-Month', 'YTD', 'March-23 TD', 'Drawdowns']

    df_perf = (df.T*100)
    df_perf = df_perf.sort_values(by=sortby, ascending=False)
    df_perf.index.name = title

    if style=='Yes':
        df_perf = df_perf.round(2).style.format('{0:,.2f}%')\
                     .background_gradient(cmap='RdYlGn')\
                     .set_properties(**{'font-size': '10pt',})

    else:
        df_perf = df_perf.round(2)
    return df_perf



def data_sov():
    #Soveriegn Fixed Income ETFs
    data_sov = yf.download('AGZ BWX EDV EMB EMLC GOVT HYD IEF IEI IGOV MUB PCY SHY SUB TFI TIP TLT TMF VWOB ZROZ', progress=False)['Adj Close']
    tickers = data_sov.columns
    data_sov.dropna(inplace=True)
    data_sov.columns = ["iShares Agency Bond ETF","SPDR  BBG Barclays International Treasury Bond ETF","Vanguard Extended Duration Treasury ETF","iShares JPM USD Emerging Markets Bond ETF","VanEck Vectors J.P. Morgan EM Local Currency Bond ETF","iShares U.S. Treasury Bond ETF","VanEck Vectors High-Yield Municipal Index ETF","iShares 7-10 Year Treasury Bond ETF","iShares 3-7 Year Treasury Bond ETF","iShares International Treasury Bond ETF","iShares National Muni Bond ETF","Invesco Emerging Markets Sovereign Debt ETF","iShares 1-3 Year Treasury Bond ETF","iShares Short-Term National Muni Bond ETF","SPDR Nuveen  BBG Barclays Municipal Bond ETF","iShares TIPS Bond ETF","iShares 20+ Year Treasury Bond ETF","Direxion Daily 20+ Year Treasury Bull 3X Shares","Vanguard Emerging Markets Government Bond ETF","PIMCO 25+ Year Zero Coupon US Treasury Index ETF"]
    return (data_sov,tickers)

def data_corp():
    #Corporate Fixed Income ETFs -  IG & HY in Developed & EM
    data_corp = yf.download('AGG ANGL BKLN BND BNDX CWB EMHY FALN FLOT FMB FPE HYEM HYG HYXE HYXU JNK LQD SHYG SRLN USIG VCIT VCLT VCSH', progress=False)['Adj Close']
    tickers = data_corp.columns
    data_corp.dropna(inplace=True)
    data_corp.columns = ["iShares Core U.S. Aggregate Bond ETF","VanEck Vectors Fallen Angel High Yield Bond ETF","Invesco Senior Loan ETF","Vanguard Total Bond Market ETF","Vanguard Total International Bond ETF","SPDR BBG Barclays Convertible Securities ETF","iShares EM High Yield Bond ETF","iShares Fallen Angels USD Bond ETF","iShares Floating Rate Bond ETF","First Trust Managed Municipal ETF","First Trust Preferred Securities & Income ETF","VanEck Emerging Markets High Yield Bond ETF","iShares USD High Yield Corporate Bond ETF","iShares USD High Yield ex Oil & Gas Corporate Bond ETF","iShares International High Yield Bond ETF","SPDR BBG Barclays High Yield Bond ETF","iShares USD IG Corporate Bond ETF","iShares 0-5 Year High Yield Corporate Bond ETF","SPDR Blackstone / GSO Senior Loan ETF","iShares Broad USD Investment Grade Bond ETF","Vanguard Intermediate-Term Corporate Bond ETF","Vanguard Long-Term Corporate Bond ETF","Vanguard Short-Term Corporate Bond ETF"]
    return (data_corp, tickers)

def data_reit(ticker='No'):
    #Real Estate Investment Trust (REIT) ETFs
    data_reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ PPTY IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS IFGL KBWY BBRE ROOF NETL STOR', progress=False)['Adj Close']['2015':]
    tickers = data_reit.columns
    data_reit.dropna(inplace=True)
    if ticker == 'Yes':
        data_reit.columns = data_reit.columns
    else:
        data_reit.columns = ['Beta Builders', 'China RE', 'NAREIT Developed Market RE', 'Quality RE', 'Hoya Housing RE', 'Europe RE', ' International Developed RE', 'Industrial RE',
                     'Premium Yield Equity RE', 'VanEck Mortgage','NetLease RE', 'Divserified RE', 'iShares Mortgage', 'iShares Residential', 'Cohen & Steers RE',
                     'Small-Cap RE', 'Super Dividend', 'Data & Infrastructure RE',
                     'Store Retail', 'Vanguard US', 'Vanguard International', 'iShares Developed Market RE']

    return (data_reit,tickers)

def data_cur():
    #Currencies
    data_cur = yf.download('KRWUSD=X BRLUSD=X IDRUSD=X MXNUSD=X RUBUSD=X CADUSD=X JPYUSD=X EURUSD=X INRUSD=X TRYUSD=X NZDUSD=X GBPUSD=X DX-Y.NYB AUDUSD=X AUDJPY=X EURCHF=X TWDUSD=X THBUSD=X COPUSD=X CNYUSD=X CLPUSD=X ZARUSD=X HKDUSD=X CHFUSD=X SGDUSD=X',progress=False)['Adj Close']['2015':]
    tickers = data_cur.columns
    data_cur.dropna(inplace=True)
    data_cur.columns = ['AUD/JPY', 'Australian Dollar', 'Brazilian Real', 'Canadian Dollar', 'Swiss Francs', 'Chilean Peso', 'Chinese Yuan',
                    'Colombian Peso', 'Dollar Index', 'EUR/CHF', 'Euro', 'British Pound', 'Hong Kong Dollar', 'Indonesian Rupiah',
                    'Indian Rupee', 'Japanese Yen', 'Korean Won', 'Mexican Peso', 'New Zealand Dollar', 'Russian Ruble',
                    'Singapore Dollar', 'Thai Baht', 'Turkish Lira', 'Taiwanese Dollar', 'South African Rand']
    return (data_cur,tickers)


def data_comd():
    #Soveriegn Fixed Income ETFs
    data_comd = yf.download('COMT GSG DBC USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F', progress=False)['Adj Close']
    tickers = data_comd.columns
    data_comd.dropna(inplace=True)
    data_comd.columns = ['Crude Oil WTI','COMT', 'Copper Miners', 'DB CMTY Fund', 'Gold Futures', 'Gold Miners',
                     'Gold ETF', 'GSCI ETF', 'Copper Futures', 'Clean Energy', 'NatGas Futures',
                     'Palladium Futures', 'Physical Palladium ETF', 'Physical Platinum ETF', 'Silver Futures', 'Silver ETF',
                     'Solar ETF', 'USO Oil ETF', 'Wheat Futures', 'Corn Futures']
    return (data_comd,tickers)

def data_country():
    #Country Equity ETFs
    data_count = yf.download('ECH EDEN EEM EEMA EEMS EEMV EFA EFAV EFG EFNL EFV EIDO EIRL EIS ENOR ENZL EPHE EPOL EPU ERUS EWA EWC EWD EWG EWGS EWH EWI EWJ EWK EWL EWM EWN EWO EWP EWQ EWS EWT EWU EWUS EWW EWY EWZ EZA EZU FXI INDY IWM SMIN SPY THD', progress=False)['Adj Close']['2015':]
    tickers = data_count.columns
    data_count.dropna(inplace=True)
    data_count.columns = ["Chile","Denmark","Emerging Markets","Asian EM","EM SmallCap","EM Low Volatility","EAFE","EAFE Low Volatility","EAFE Growth","Finland","EAFE Value","Indonesia","Ireland","Israel","Norway","New Zealand","Philippines","Poland","Peru","Russia","Australia","Canada","Sweden","Germany","Germany Small Cap","Hong Kong","Italy","Japan","Belgian","Switzerland","Malaysia","Netherlands","Austria","Spain","France","Singapore","Taiwan","UK","UK Small Cap","Mexico","South Korea","Brazil","South Africa","EuzoZone", "China","India", "US Russell 2000", "India Small Cap", "US S&P500", "Thailand"]
    return (data_count,tickers)


def data_equities():
    data_eq = yf.download("ARKK BLOK BOTZ CHIE CHIH CHII CHIK CHIM CHIQ CHIR CHIS CHIU CHIX FXI INTF JETS LRGF MTUM QFN.AX QQQ QRE.AX QUAL SIZE SPY STK.PA STN.PA STP.PA STQ.PA STR.PA STS.PA STU.PA STW.PA STZ.PA USMV VLUE VYM XAR XBI XEG.TO XES XFN.TO XHB XHE XHS XIT.TO XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XMA.TO XME XOP XPH XRT XSD XST.TO XSW XTN XUT.TO", progress=False)['Adj Close']['2018':]
    tickers = data_eq.columns
    data_eq.dropna(inplace=True)
    data_eq.columns = ["U.S. Innovation ETF","U.S. Blockchain ETF","U.S. Robotics&AI ETF","China Energy ETF","China Health Care ETF","China Industrial ETF","China Technology ETF","China Materials ETF","China Consumer Cyclical ETF","China Real Estate ETF","China Consumer Non-cyclical ETF","China Utilities ETF","China Finance ETF","China  ETF","DM Multi Factor ETF","U.S. Airlines ETF","U.S. Multi Factor ETF","U.S. Momentum ETF","Australia Finance ETF","US NASDAQ","Australia Resources ETF","U.S. Quality ETF","U.S. Size ETF","US S&P500 ETF","Europe Technology ETF","Europe Energy ETF","Europe Materials ETF","Europe Industrial ETF","Europe Consumer Cyclical ETF","Europe Consumer Non-cyclical ETF","Europe Utilities ETF","Europe Healthcare ETF","Europe Finance ETF","U.S. Low Volitality ETF","U.S. Value ETF","U.S. High Dividend Yield ETF","U.S. Aerospace&Defence ETF","U.S. Biotech ETF","Canada Energy ETF","U.S. Oil & Gas Equipment & Services ETF","Canada Finance ETF","U.S. Homebuilding ETF","U.S. Health Care Equipment & Supplies ETF","U.S. Health Care Services ETF","Canada Technology ETF","U.S. Basic Materials ETF","U.S. Telecom ETF","U.S. Energy ETF","U.S. Finance ETF","U.S. Industrial ETF","U.S. Technology ETF","U.S. Consumer Non-cyclical ETF","U.S. Real Estate ETF","U.S. Utilities ETF","U.S. Healthcare ETF","U.S. Consumer Cyclical ETF","Canada Materials ETF","U.S. Metals & Mining ETF","U.S. Oil&Gas Exploration ETF","U.S. Pharma ETF","U.S. Retail ETF","U.S. Semi Conductors ETF","Canada Consumer Non-cyclical ETF","U.S. Software ETF","U.S. Transportation ETF","Canada Utilities ETF"]
    return (data_eq, tickers)



def heatmap(rets, title='Cross Asset ETFs Heatmap', figsize=(15,10), annot_size=12, n_rows=10, n_cols=8, pct='Yes', mon=None):

    rets.columns = ['Return']
    rets = rets.sort_values('Return', ascending=False)
    symbols = (np.asarray(list(rets.index))).reshape(n_rows,n_cols)
    pct_rets = (np.asarray(rets['Return'].values)).reshape(n_rows,n_cols)
    rows =[]
    for i in range(1,n_rows+1):
        rows += list(np.repeat(i,n_cols))

    cols = list(list(np.arange(1,n_cols+1))*n_rows)
    rets['Rows'] = rows
    rets['Cols'] = cols

    result = rets.pivot(index = 'Rows', columns = 'Cols', values = 'Return')
    if pct=='Yes':
        labels = (np.asarray(["{0} \n {1:.2%} ".format(symb,value)
                     for symb, value in zip(symbols.flatten(), pct_rets.flatten())])).reshape(n_rows,n_cols)

    else:
        labels = (np.asarray(["{0} \n {1:.2f} \n {2}".format(symb,value, mon)
                     for symb, value, mon in zip(symbols.flatten(), pct_rets.flatten(), mon.flatten())])).reshape(n_rows,n_cols)
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title, fontsize=15)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return sns.heatmap(result, annot=labels, fmt="", cmap = 'RdYlGn', linewidth=0.30, ax=ax, annot_kws={"size": annot_size})


def alok_heatmap():
    alok_data = yf.download("CADUSD=X EURUSD=X GBPUSD=X USDINR=X USDSGD=X ACWI URTH ^NDX ^NSEI ^GSPC BZ=F DBC GLD GC=F PALL USO O9P.SI ANGL EDV EMB EMHY EMLC FPE GOVT HYEM HYG JNK LQD SHYG TLT TMF VWOB ZROZ ABR A17U.SI AV.L J85.SI J91U.SI MERY.PA ME8U.SI RGL.L VNQ VNQI 8697.T AAPL ABT ADBE AHCO AIEQ AIIQ AMD AMZN B3SA3.SA BABA BB BBJP BOX CLX CMG CRWD CTXS DBX DOCU DPZ ERUS EWT EWZ FXI GILD GOOG HD JD LSE.L MA MFA MOEX.ME MSFT NDAQ NEM NFLX NLOK NTES NVDA PTON PYPL PZZA QQQ S68.SI SHOP SMG SNY SPY SQ TDOC TMO TWLO TWOU TWTR V WORK X.TO ZM PD MKC NKE LQDA.L UST.PA PSHZF VTWO IWM", start='2019-12-31',end=date.today())['Adj Close']

    alok_data = alok_data.rename(columns = {'^NSEI':'NIFTY50', 'BZ=F':'Crude', 'A17U.SI':'AREIT', 'AV.L':'AV', 'J85.SI':'CDREIT',
                                            'J91U.SI':'EREIT','S68.SI':'SGX', 'X.TO':'TMX Group', 'CADUSD=X':'CADUSD','ME8U.SI':'MINT',
                                            'EURUSD=X':'EURUSD', 'GBPUSD=X':'GBPUSD', 'USDINR=X':'USDINR', 'USDSGD=X':'USDSGD',
                                            '^NDX':'NASDAQ', 'URTH':'MSCI Wrld', '^GSPC':'S&P500', 'GC=F': 'GOLD', 'O9P.SI':'AHYG'})
    alok_data = alok_data.ffill().asfreq('B').dropna()
    alok_data1 = pd.DataFrame(data = (alok_data.iloc[-1,:],alok_data.pct_change().iloc[-1,:])).T
    alok_data1.columns = ['Price', 'Chg (%)']
    alok_data2 = alok_data1.style.format({'Chg (%)': "{:.2%}", 'Price': "{:.2f}"})
    alok_map = returns_heatmap_alok(alok_data, drawdowns2020(alok_data), title='Securities', style='No')
    prices = pd.DataFrame(alok_data1['Price'])
    prices.index.name = 'Securities'
    final = prices.merge(alok_map, on='Securities').sort_values(by='1-Day', ascending=False).style.format('{0:,.2f}%', subset=list(alok_map.columns))\
                     .format('{0:,.2f}', subset=['Price'])\
                     .background_gradient(cmap='RdYlGn', subset=list(alok_map.columns))\
                     .set_properties(**{'text-align':'left','font-family': 'Segoe UI','font-size': '10.5px'})
    return (final, alok_data1, alok_data)


def cross_asset_data():
    """
    """
    data_sov = yf.download('SHY IEF TLT IEI EMB EMLC AGZ BWX TIP', progress=False)['Adj Close']['2019':]
    data_sov.dropna(inplace=True)
    data_corp = yf.download('AGG BND BNDX LQD HYG SHYG JNK FALN ANGL FPE HYXE HYXU HYEM EMHY', progress=False)['Adj Close']['2019':]
    data_corp.dropna(inplace=True)
    data_reit = yf.download('VNQ VNQI SRVR INDS HOMZ REZ IFEU REM MORT SRET RFI FFR GQRE CHIR FFR WPS PPTY IFGL KBWY ROOF NETL SPG SKT STOR', progress=False)['Adj Close']['2019':]
    data_reit.dropna(inplace=True)
    data_cur = yf.download('KRWUSD=X  BRLUSD=X  IDRUSD=X  MXNUSD=X  RUBUSD=X  CADUSD=X  JPYUSD=X  EURUSD=X  INRUSD=X  TRYUSD=X  NZDUSD=X  GBPUSD=X  DX-Y.NYB  AUDUSD=X  AUDJPY=X  EURCHF=X', progress=False)['Adj Close']['2019':].iloc[:-1,:]
    data_cur.dropna(inplace=True)
    data_comd = yf.download('COMT USO CL=F HG=F COPX GC=F GLD GDX PA=F PALL PPLT SI=F SIL ICLN TAN W=F ZC=F NG=F', progress=False)['Adj Close']['2019':]
    data_comd.dropna(inplace=True)

    all_data = data_sov.merge(data_corp, on='Date').merge(data_reit, on='Date').merge(data_cur, on='Date').merge(data_comd, on='Date')
    return all_data


def cross_asset_heatmap(data, n_rows, n_cols, days=1, figsize=(15,10), annot_size=12, title='Cross Asset ETFs Heatmap'):
    """
    """
    rets = pd.DataFrame(data.pct_change(days).iloc[-1,:])
    return heatmap(rets, n_rows=n_rows, n_cols=n_cols, figsize=figsize, annot_size=annot_size, title=title)


#nbi:hide_in
def import_data(asset_class, sortby='1-Day'):
    """

    """
    tdy = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year)
    oneyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-1)
    threeyr = str(date.today().day)+'/'+str(date.today().month)+'/'+str(date.today().year-3)
    #Import list of ETFs and Ticker Names
    etf_list = pd.read_excel('etf_names.xlsx', header=0, sheet_name=asset_class)

    #Define function to fetch historical data from Investing.com
    def hist_data(name, country):
        df = investpy.get_etf_historical_data(etf=name, country=country, from_date=oneyr, to_date=tdy)['Close']
        df = pd.DataFrame(df)
        df.columns = [name]
        return df

    #Build an empty df to store historical 1 year data
    df = pd.DataFrame(index=pd.bdate_range(start=oneyr, end=date.today()))
    df.index.name='Date'

    #download and merge all data
    for i in range(len(etf_list)):
            df = df.join(hist_data(etf_list[asset_class][i], etf_list['Country'][i]), on='Date')

    #Forward fill for any missing days i.e. holidays
    df1 = df.iloc[:-1,:].ffill().dropna()
    df1.index.name = 'Date'


    #Generate multi timeframe returns table
    df0 = pd.DataFrame(data = (df1.pct_change(1).iloc[-1,:], df1.pct_change(5).iloc[-1,:], df1.pct_change(21).iloc[-1,:], df1.pct_change(63).iloc[-1,:], df1['2020':].iloc[-1,:]/df1['2020':].iloc[0,:]-1, df1['2020':].iloc[-1,:]/df1['2020-03-23':].iloc[0,:]-1, df1.pct_change(126).iloc[-1,:], df1.pct_change(252).iloc[-1,:], drawdowns2020(df1)))
    df0.index = ['1-Day', '1-Week', '1-Month', '3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns']

    df_perf = (df0.T*100)
    df_perf.index.name = asset_class


    #Add Ticker Names and sort the dataframe
    tickers = pd.DataFrame(etf_list['Ticker'])
    tickers.index = etf_list[asset_class]
    df2 = tickers.merge(df_perf, on=asset_class)
    df2  = df2.sort_values(by=sortby, ascending=False)
    df2 = df2.round(2).style.format('{0:,.2f}%', subset=list(df2.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(df2.drop(['Ticker'], axis=1).columns))\
                     .set_properties(**{'font-size': '10pt',})

    return df2, df1




def import_heatmap(data, sortby):
    """
    data refers to return data for multiple timeframes
    sortby requires a str format input which includes 1-Day, 1-Week, 1-Month, YTD, 1-Year
    """
    #Sort values
    data = data.sort_values(by=sortby, ascending=False)

    #Conditionally format the dataframe
    data = data.round(2).style.format('{0:,.2f}%', subset=list(data.drop(['Ticker'], axis=1).columns))\
                     .background_gradient(cmap='RdYlGn', subset=(data.drop(['Ticker'], axis=1).columns))\
                     .set_properties(**{'font-size': '10pt',})
    return data


def analytics(price_data, returns_data):
    """

    """
    def heatmap_all(sortby):
        return import_heatmap(returns_data, sortby=sortby)

    interact(heatmap_all,
             sortby = widgets.Dropdown(options=['1-Day', '1-Week', '1-Month','3-Month', 'YTD', 'March-23 TD', '6-Month', '1-Year', 'Drawdowns'], value='1-Day', description = 'Sort By: '));

    def perf_chart_all(start_date):
        df = ((((1+price_data.dropna()[start_date:date.today()].pct_change().fillna(0.00))).cumprod()-1)).round(4)
        fig = px.line(df, x=df.index, y=df.columns)
        fig.update_layout(title = price_data.columns.name + ' ETFs',
                       xaxis_title='Date',
                       yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                       legend_title_text='ETFs', plot_bgcolor = 'White', yaxis_tickformat = '%')
        fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
        fig.update_yaxes(automargin=True)
        fig.show()


    interact(perf_chart_all,
             start_date = widgets.Dropdown(options=[('1 Month', one_m), ('3 Months', three_m), ('6 Month', six_m),
                                                ('YTD', ytd), ('1 Year', one_yr),
                                                ('From 23rd March', '2020-03-23')], value=ytd, description = 'Period'));


    #nbi:hide_in
    def trend_analysis(start_date,inv, ma):
        d = (price_data.pct_change(ma).dropna()[start_date:date.today()].resample(inv).agg(lambda x: (x + 1).prod() - 1).round(4)*100)
        fig = go.Figure(data=go.Heatmap(
                z=((d - d.mean())/d.std()).round(2).T.values,
                x=((d - d.mean())/d.std()).index,
                y=list(price_data.columns), zmax=3, zmin=-3,
                colorscale='rdylgn', hovertemplate='Date: %{x}<br>ETF: %{y}<br>Return Z-Score: %{z}<extra></extra>', colorbar = dict(title='Return Z-Score')))

        fig.update_layout(
            title= price_data.columns.name + ' ETFs Return Trend Analysis',
            xaxis_nticks=20, font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"))

        fig.show()

    style = {'description_width': 'initial'}
    interact(trend_analysis,
             start_date = widgets.Dropdown(options=[('1 Month', one_m), ('3 Months', three_m), ('6 Months', six_m),
                                                    ('YTD', ytd), ('1 Year', one_yr),
                                                    ('From 23rd March', '2020-03-23')], value=ytd, description = 'Period'),
             inv = widgets.Dropdown(options=[('Daily', 'B'), ('Weekly', 'W'), ('Monthly', 'BM')], value='B', description='Return Interval:', style=style),
             ma = widgets.BoundedIntText(value=15, min=1, max=30, step=1, description='Rolling Return Period:', disabled=False, style=style));
