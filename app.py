import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Equity Signal Lab | ACCT 3343', layout='wide')
st.title('Equity Signal Lab')
st.caption('ACCT 3343 | Independent US equity screening | Fundamental signals, not investment advice')
DATA = Path(__file__).with_name('universe.csv')
if not DATA.exists():
    st.error('universe.csv is missing. Run build_universe.py first, then upload the resulting CSV beside app.py.')
    st.stop()
@st.cache_data
def load():
    d=pd.read_csv(DATA)
    required=['ticker','name','sector','market_cap','total_assets','net_income','operating_cash_flow','gross_profit','revenue','total_liabilities','current_assets','current_liabilities','retained_earnings','ebit','book_equity','operating_assets','operating_liabilities','prior_operating_assets','prior_operating_liabilities','prior_total_assets','capital_expenditure']
    missing=[c for c in required if c not in d]
    if missing: raise ValueError('Missing CSV columns: '+', '.join(missing))
    d=d.drop_duplicates('ticker').copy()
    for c in required[3:]: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d
try: df=load()
except Exception as e:
    st.error(str(e));st.stop()

def div(a,b):
    return a.div(b.where(b>0)).replace([np.inf,-np.inf],np.nan)

df['STA']=div(df.net_income-df.operating_cash_flow,df.total_assets)
df['SNOA']=div(df.operating_assets-df.operating_liabilities,df.total_assets)
df['delta_NOA']=div((df.operating_assets-df.operating_liabilities)-(df.prior_operating_assets-df.prior_operating_liabilities),(df.total_assets+df.prior_total_assets)/2)
df['GPA']=div(df.gross_profit,df.total_assets)
df['BM']=div(df.book_equity,df.market_cap)
df['FCF']=df.operating_cash_flow-df.capital_expenditure.abs()
df['FCF_yield']=div(df.FCF,df.market_cap)
# Altman original publicly traded manufacturing model. Not used in ranking for other sectors.
df['working_capital']=df.current_assets-df.current_liabilities
df['Altman_Z']=1.2*div(df.working_capital,df.total_assets)+1.4*div(df.retained_earnings,df.total_assets)+3.3*div(df.ebit,df.total_assets)+0.6*div(df.market_cap,df.total_liabilities)+div(df.revenue,df.total_assets)
df.loc[~df.sector.eq('Industrials'),'Altman_Z']=np.nan

st.sidebar.header('Universe & screening')
sectors=sorted(df.sector.dropna().unique())
chosen=st.sidebar.multiselect('Sectors',sectors,default=sectors)
cap=st.sidebar.slider('Minimum market cap ($B)',0.0,50.0,0.3,0.1)
sta_pct=st.sidebar.slider('SELL: accrual percentile',50,95,75)
snoa_pct=st.sidebar.slider('SELL: operating assets percentile',50,95,75)
min_flags=st.sidebar.slider('SELL: minimum red flags',1,3,2)
use_z=st.sidebar.checkbox('Include Altman distress flag for industrials',value=True)
z_cut=st.sidebar.slider('Altman Z distress cutoff',0.5,3.5,1.81,0.01,disabled=not use_z)
min_gpa=st.sidebar.slider('BUY: minimum gross profitability',0.0,0.8,0.10,0.01)
min_fcf=st.sidebar.slider('BUY: minimum FCF yield',-0.1,0.2,0.0,0.005,format='%.3f')
weight=st.sidebar.slider('BUY: profitability ranking weight',0,100,50,5)
exclude_ticker=st.sidebar.text_input('Exclude your 60-second pitch ticker(s), comma-separated','').upper()
excluded={s.strip() for s in exclude_ticker.split(',') if s.strip()}
d=df[df.sector.isin(chosen)&(df.market_cap>=cap*1e9)&~df.ticker.isin(excluded)].copy()
# Percentile thresholds recompute within the currently selected universe.
sta_threshold=d.STA.quantile(sta_pct/100)
snoa_threshold=d.SNOA.quantile(snoa_pct/100)
d['flag_accrual']=d.STA.notna()&(d.STA>=sta_threshold)
d['flag_noa']=d.SNOA.notna()&(d.SNOA>=snoa_threshold)
d['flag_distress']=use_z&d.Altman_Z.notna()&(d.Altman_Z<z_cut)
d['red_flags']=d[['flag_accrual','flag_noa','flag_distress']].sum(axis=1)
d['sell_score']=d.STA.rank(pct=True)*0.5+d.SNOA.rank(pct=True)*0.5
sell=d[(d.red_flags>=min_flags)&d[['STA','SNOA']].notna().all(axis=1)].sort_values(['red_flags','sell_score'],ascending=False)
# BUY uses distinct measures from SELL. Restrict B/M to positive equity and positive FCF.
d['gpa_rank']=d.GPA.rank(pct=True)
d['bm_rank']=d.BM.where(d.BM>0).rank(pct=True)
d['fcf_rank']=d.FCF_yield.rank(pct=True)
d['buy_score']=(weight/100)*d.gpa_rank+(1-weight/100)*(0.5*d.bm_rank+0.5*d.fcf_rank)
buy=d[(d.GPA>=min_gpa)&(d.FCF_yield>=min_fcf)&(d.BM>0)&d[['GPA','BM','FCF_yield','buy_score']].notna().all(axis=1)].sort_values('buy_score',ascending=False)
cols=st.columns(4)
for c,label,val in zip(cols,['Original universe','Companies screened','SELL candidates','BUY candidates'],[len(df),len(d),len(sell),len(buy)]): c.metric(label,val)
if len(df)<500: st.warning('The source CSV contains fewer than 500 companies. The broad-universe requirement is not yet met.')
st.caption(f'Data file: {DATA.name}. Percentile thresholds are calculated on the filtered universe; missing values are not imputed. SELL and BUY lists may overlap because they measure different characteristics.')
t1,t2,t3,t4=st.tabs(['SELL screen','BUY screen','Company drill-down','Methodology & conclusions'])
with t1:
    st.subheader('Earnings quality & financial red flags')
    st.dataframe(sell[['ticker','name','sector','STA','SNOA','Altman_Z','red_flags','sell_score']],use_container_width=True,hide_index=True)
    if not sell.empty:
        fig=px.scatter(sell,x='STA',y='SNOA',hover_name='ticker',color='red_flags',title='SELL candidates: accruals vs operating assets')
        st.plotly_chart(fig,use_container_width=True)
with t2:
    st.subheader('Quality and valuation opportunities')
    st.dataframe(buy[['ticker','name','sector','GPA','BM','FCF_yield','buy_score']],use_container_width=True,hide_index=True)
    if not buy.empty:
        fig=px.scatter(buy,x='BM',y='GPA',size='market_cap',hover_name='ticker',color='buy_score',title='BUY candidates: value vs profitability')
        st.plotly_chart(fig,use_container_width=True)
with t3:
    if len(d):
        ticker=st.selectbox('Choose a company',sorted(d.ticker.tolist()))
        r=d.set_index('ticker').loc[ticker]
        st.subheader(f"{r['name']} ({ticker})")
        st.write(f"Sector: {r['sector']} | Market capitalization: ${r['market_cap']/1e9:,.2f}B")
        st.dataframe(pd.DataFrame({'Metric':['Scaled total accruals','Scaled net operating assets','Change in net operating assets','Gross profitability','Book-to-market','Free cash flow yield','Altman Z (industrials only)','SELL red flags','BUY composite score'],'Value':[r.STA,r.SNOA,r.delta_NOA,r.GPA,r.BM,r.FCF_yield,r.Altman_Z,r.red_flags,r.buy_score]}),hide_index=True,use_container_width=True)
with t4:
    st.subheader('Data and formulas')
    st.markdown('''**Universe:** Independently sourced US-listed equities on NYSE/Nasdaq; financial sector excluded in the data builder. This is a snapshot, not live market data.  
**STA:** (net income − operating cash flow) / total assets. Higher values can signal weaker earnings quality.  
**SNOA:** (operating assets − operating liabilities) / total assets. Higher levels may indicate balance-sheet accrual concerns.  
**GPA:** gross profit / total assets. Higher is better.  
**B/M:** book equity / market capitalization. Higher indicates cheaper book valuation; negative equity excluded from BUY.  
**FCF yield:** (operating cash flow − absolute capital expenditures) / market capitalization.  
**Altman Z:** original public-manufacturing equation; used only as an optional approximate distress flag for industrial firms. Industries are broader than manufacturing, so interpret cautiously.  
**SELL:** at least N of the accrual, SNOA, and optional distress flags; ordered by flag count and average accrual/SNOA percentile.  
**BUY:** positive book equity, threshold profitability and FCF yield; weighted percentile score using GPA, B/M, and FCF yield.  
**Missing data:** never filled with zero; firms with incomplete measures may not qualify.  
**Limitations:** Yahoo statement classifications, fiscal-year timing, and market-value dates may differ. Rankings are screening signals, not intrinsic valuations or guaranteed returns.''')
    st.subheader('Investment conclusions — complete after the actual CSV is verified')
    st.info('For your submission, replace this box with your own written investment cases for at least one top SELL and one top BUY company. Cite their actual metrics, explain the economic logic, risks, and why they rank above peers. Do not submit with this placeholder.')
    st.write('Current top SELL tickers:',', '.join(sell.ticker.head(3)) or 'None under selected filters')
    st.write('Current top BUY tickers:',', '.join(buy.ticker.head(3)) or 'None under selected filters')
