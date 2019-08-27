import datetime
import pandas as pd
import numpy as np
import json
#-------------------- utils -------------------------------------
def get_trade_days(end_date,count):
    
    date_list = pd.read_csv('../data/选股原始指标/is_St.csv',index_col=0).index.tolist()
    # print(date_list[-1])
    return date_list[date_list.index(end_date)-count:date_list.index(end_date)+1]
    
def get_pctd(date, count=1):
    df = pd.read_pickle('../data/factors/pctd'+str(count)+'.pkl')
    return df.loc[date].sort_values().index.tolist()

def get_pctd60_rank(date):
    df = pd.read_pickle('../data/factors/pctd60_rank.pkl')
    return df.loc[date].dropna().sort_values().index.tolist()

def get_pctd1_rank(date):
    df = pd.read_pickle('../data/factors/pctd_rank.pkl')
    return df.loc[date].dropna().sort_values().index.tolist()   

def get_all_securities(types,date):
    df = pd.read_pickle('../data/allStock.pkl')
    df = df[(df['end_date']>date) & (date>df['start_date'])]
    return df.index.tolist()

def get_all_securities_df(date,types=['stock']):
    return pd.read_pickle('../data/allStock.pkl')

def get_mkt_cap(date,count=1):
    df = pd.read_pickle('../data/factors/mkt_cap.pkl') 
    return df.loc[date].dropna().sort_values().index.tolist()

def get_mkt_cap_ind(date,count=1):
    df = pd.read_pickle('../data/factors/mkt_cap_ind.pkl') 
    return df.loc[date].dropna().sort_values().index.tolist()

def get_pe_negative(date, count=1):
    return []

def get_is_st(date, count=1):
    df = pd.read_pickle('../data/isSt.pkl')
    return df.loc[date][df.loc[date] == 1].index.tolist()

def get_is_susp(date, count=1):
    df = pd.read_pickle('../data/isSusp.pkl')
    return df.loc[date][df.loc[date] == 1].index.tolist()

def get_susp_sum(date, count=60, threshold=10):
    df = pd.read_pickle('../data/isSusp.pkl')
    id = df.index.tolist()
    end = id.index(date)
    start = end-count
    return df.loc[start:end].sum()[df.loc[start:end].sum()>threshold].tolist()

def get_price(universe, end_date,count=None,start_date=None,fields='close'):
    if universe==500:
        df = pd.read_pickle('../data/中证500/500pctd.pkl')
        try:
            date_list = get_trade_days(end_date,count)
            df = df.loc[date_list]
            return df
        except:
            pass
        return df
    if fields == 'close':
        date_list = get_trade_days(end_date,count)
        return pd.read_pickle('../data/factors/preclosed.pkl').loc[date_list]

    return

def get_all_industry():
    return pd.read_pickle('../data/industry.pkl')

def get_return_openam10():
    return pd.read_pickle('../data/factors/return_openam10.pkl')

def get_return_closeam10():
    return pd.read_pickle('../data/factors/return_closeam10.pkl')

def get_ind_rate():
    return pd.read_pickle('../data/ind_rate.pkl')

def get_ind_dict():
    with open('../data/ind_dict.json','r') as ind_dict_json:
        ind_dict = json.load(ind_dict_json)
    return ind_dict

def get_univ_dict(start_date, end_date):
    df_univ = pd.read_pickle('../data/univ_dict.pkl')
    d = df_univ.loc[:,df_univ.columns[df_univ.columns.tolist().index(start_date): \
                df_univ.columns.tolist().index(end_date)+1]].to_dict('list')

    for k in d:
        d[k] = list(set(d[k]))
    return d
#-------------------- 待完善 ---------------------------------------