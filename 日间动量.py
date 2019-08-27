import datetime
import pandas as pd
import numpy as np
from data import * 
from factor import *
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as st
import scipy.stats as st
import pickle
'''
    因子计算-底层方程
'''
# -----------------Step1：准备数据------------------

def get_trade_dates(end, count=980, interval=10):
    '''
        按调仓周期时间节点返回交易日序列列表
    '''
    date_list = list(get_trade_days(end_date=end,count=count))
    date_list = date_list[::-1]
    date_list = list(filter(lambda x: date_list.index(x)%interval==0, date_list))
    date_list=date_list[::-1]
    return date_list


def get_stock_pool(date, index='all'):
    '''
        获取当天符合交易条件的股票池
    '''
    df = get_all_securities_df(types=['stock'],date=date)
    # 筛选条件1：上市半年（120个交易日）以上
    dayBefore = get_trade_days(end_date=date, count=60)[0]
    # print(date, dayBefore)
    df = df[(df['start_date']<dayBefore) & (df['end_date']>date)]
    universe_pool = list(df.index)
    # 筛选条件2：剔除前60天涨幅前30%，后20%
    pctd60 = get_pctd60_rank(date)
    pctd60_top30_bot20 = pctd60[:int(len(pctd60)*0.2)] + \
                    pctd60[-int(len(pctd60)*0.3):]

    # 筛选条件3：剔除pe<0, 
    pe_negative = get_pe_negative(date, count=1)
    
    # 筛选条件4：剔除前一天跌幅最大20%
    pctd1 = get_pctd1_rank(date)
    pctd1_bot20 = pctd1[-int(len(pctd60)*0.2):]

    # 筛选条件5：剔除ST股
    is_st = get_is_st(date, count=1)

    # 筛选条件6：剔除流动市值最大的10%和最小的10%
    mkt_cap = get_mkt_cap_ind(date, count=1)
    mkt_cap_top10_bot10 = mkt_cap[:int(len(mkt_cap)*0.1)] + \
                        mkt_cap[-int(len(mkt_cap)*0.1):]
    
    #  筛选条件7：剔除最近三个月（60个交易日）是否停牌超过10天，且上一交易日停牌
    is_suspd1 = get_is_susp(date, count=1)
    susp_sumd60 = get_susp_sum(date, count=60, threshold=10)

    filters = list(set(pctd60_top30_bot20 + pe_negative + pctd1_bot20 + is_st +\
                     mkt_cap_top10_bot10 + is_suspd1 + susp_sumd60))
    stock_pool = [stock for stock in universe_pool if stock not in filters ] 
    return stock_pool

def get_stock_universe(trade_days_list, index='all'):
    '''
        获取交易日序列，符合条件股票池的列表
    '''
    univ_list = []
    for date in trade_days_list:
        stock_pool=get_stock_pool(date,index)
        univ_list.append(stock_pool)    
    return univ_list

def get_return(trade_date_list, count=980):
    '''
        dev@获取按调仓间隔的收益序列，获取逐日(后一天)的收益序列
    '''
    date = max(trade_date_list)
    universe = get_stock_pool(date, index='all')
    # 获取交易日期内前收盘价数据
    price = get_price(universe, end_date=date,count=count,fields='close') 
    return_df = price.loc[trade_date_list].pct_change().shift(-1)
    price_index = price.index.tolist()
    new_index = price_index[price_index.index(trade_date_list[0]):]
    all_return_df=price.loc[new_index].pct_change().shift(-1)
    return return_df, all_return_df

def get_factor_by_day(date, g_factor_list, g_index='all'):
    '''
        获取该交易日的因子值，存储在字典中
    '''
    # factor_dict的key是因子名字，value是dataframe，其中行为日期，列为股票
    factor_list = g_factor_list
    index = g_index
    if index == 'all':
        universe = get_all_securities(types=['stock'],date=date)
    else: ### to do list
        # universe = get_index_stocks(index, date=date)
        pass
    factor_dict = calc_factors(universe,factor_list,date)
    return factor_dict

def get_Industry_by_day(date): 
    # 返回日期我index,股票代码为columns，value为行业的df
    pass


def dateTransform(date_list):
    '''
        date_list由int转化为datetime
    '''
    date_list_str=map(lambda x: str(int(x)),date_list)
    date_list_datetime=map(lambda x:datetime.datetime.strptime(x,'%Y%m%d'),date_list_str)
    return list(date_list_datetime)

# -----------------Step2：处理数据-----------------------------
def rank(se):
    return se.rank()

def pretreat_factor(factor_df,univ_dict,factor_name):
    pretreat_factor_df=pd.DataFrame()
    danger_list=[]
    # print(factor_df.loc[list(univ_dict.keys())].isnull().sum(axis=1))
    for date in list(univ_dict.keys()):     #循环从这儿开始
        # 把该日的因子，行业，市值数据取好。
        univ=univ_dict[date]
        print(len(univ))
        factor_se=factor_df.loc[date,univ]
        # factor_se=factor_df.loc[date,univ].dropna()
        
        # market_cap_se=MC_df.loc[date,stock_list]
        # industry_se=all_industry_df.loc[0]
        # 进行数据处理
        # factor_se=winsorize(factor_se)
        # factor_se=neutralize(factor_se,industry_se,market_cap_se)
        factor_se=rank(factor_se)
        # 把中性化的数据赋值
        # factor_se_withnan[factor_se.index]=factor_se
        print(date)
        pretreat_factor_df=pretreat_factor_df.append(factor_se.to_frame(date).T)

        danger=np.isnan(factor_se).sum()/len(factor_se)
        danger_list.append(danger)
    print('pretreat_factor_df null sum',pretreat_factor_df.isnull().sum(axis=1))
    return pretreat_factor_df,danger_list

def replace_nan_indu(all_industry_df,factor_df,univ_dict):
    fill_factor=pd.DataFrame()
    for date in list(univ_dict.keys()):
        # 调仓日
        univ=univ_dict[date]
        # date日的因子值
        factor_by_day=factor_df.loc[date,univ].to_frame('values')
        industry_by_day=all_industry_df.loc[date,univ].dropna().to_frame('industry')
        factor_by_day=factor_by_day.merge(industry_by_day,left_index=True,right_index=True,how='inner')
        mid=factor_by_day.groupby('industry').median()
        factor_by_day=factor_by_day.merge(mid,left_on='industry',right_index=True,how='left')
        factor_by_day.loc[pd.isnull(factor_by_day['values_x']),'values_x']=factor_by_day.loc[pd.isnull(factor_by_day['values_x']),'values_y']
        fill_factor=fill_factor.append(factor_by_day['values_x'].to_frame(date).T)
    return fill_factor

# -----------------Step3: 因子计算中层方程 --------------------
def prepareData(trade_date_list, g_factor_list, g_count, filter_factor_list, g_index='all'):
    # print('1.1正在汇总股票...')
    # univ_list = get_stock_universe(trade_date_list,g_index)

    print('1.2正在汇总回报...')
    return_df, all_return_df = get_return(trade_date_list, g_count)

    print('1.3正在汇总因子字典...')
    # pool = ThreadPool(processes=16)
    # # 因子对应当日value字典的时间序列列表
    # frame_list = pool.map(get_factor_by_day, trade_date_list)
    # pool.close()
    # pool.join()
    all_factor_dict={}


    for fac in g_factor_list:
        # 单个因子对应时间index,股票池columns的value
        # y = [x[fac.name] for x in frame_list]
        # y = pd.concat(y, axis=0)
        all_factor_dict[fac.name] = fac.calc()

    print('1.4计算过滤因子...')
    # pool=ThreadPool(processes=16)
    # frame_list=pool.map(get_Industry_by_day,trade_date_list)
    # pool.close()
    # pool.join()
    # all_industry_df=get_all_industry()
    filter_factor_dict = {}
    for fac in filter_factor_list:
        filter_factor_dict[fac.name] = fac.calc()
    print('完成') 
    univ_dict = get_univ_dict(trade_date_list[0],trade_date_list[-1]) 
    # univ_dict = dict({})
    # for i in range(len(return_df)):
    #     univ_dict[return_df.index[i]] = univ_list[i]
    
    return univ_dict,return_df,all_return_df,all_factor_dict, filter_factor_dict

def TrimData(univ_dict, all_factor_dict, filter_factor_dict, is_filter=False):
    i=1
    new_all_factor_dict={}
    print('\n修理因子数据进度')
    for factor in all_factor_dict:
        factor_df=all_factor_dict[factor]
        #2.1 把nan用行业中位数代替，依然会有nan，比如说整个行业没有该项数据，或者该行业仅有此一只股票，且为nan。
            # factor_df=replace_nan_indu(all_industry_df,factor_df,univ_dict)
        #2.2 去极值、中性化、标准化，上述的nan依然为nan。
        factor_df,danger_list=pretreat_factor(factor_df, univ_dict,factor)
        new_all_factor_dict[factor]=factor_df
        print("\n%.2f %%" %(i/len(list(all_factor_dict.keys()))*100)) 
        i=i+1
        if max(danger_list)>0.05:
            print("\ndangerous factor %s %f %f" % (factor,min(danger_list),max(danger_list)),end=',')  

    print('\n修理过滤因子数据进度')
    i=0
    new_filter_factor_dict = {}
    if is_filter == True:
        for factor in filter_factor_dict:
            factor_df = filter_factor_dict[factor]
            factor_df, danger_list = pretreat_factor(factor_df, univ_dict, factor)
            new_filter_factor_dict[factor] = factor_df
            print("\n%.2f %%" %(i/len(list(filter_factor_dict.keys()))*100)) 
            i=i+1
            if max(danger_list)>0.05:
                print("\ndangerous factor %s %f %f" % (factor,min(danger_list),max(danger_list)),end=',')  

    return new_all_factor_dict, new_filter_factor_dict

# -----------------Step4: 程序运行 --------------------
if __name__ == 'main':
    file_suffix = str(input('please input suffix of pkl file:'))
    hold_days = int(input('please hold days:'))
    # global g_index
    # global g_factor_list
    # global g_count
    # global g_univ_dict
    filter_factor_list = [alphad1(),alphad2(),alphad3(),alphad4(),alphad5(),alphad10()]
    g_univ_dict=0
    g_index='all'
    if 'd60_d240' in file_suffix:
        g_factor_list=[alphad60(),alphad120(),alphad240()]
    elif 'd1_d5' in file_suffix:
        g_factor_list=[alphad1(),alphad2(),alphad3(),alphad4(),alphad5()]
    elif 'd10_d30' in file_suffix:
        g_factor_list=[alphad10(),alphad20(),alphad30()]  
    else:
        print('factors is: {}'.format(file_suffix.split('_hold')[0]))
        factor = locals()[file_suffix.split('_hold')[0]+'()']
        g_factor_list = [Factor]      
    g_count=980


    # 获取当前日期
    # today=int(str(datetime.date.today()).replace('-',''))

    # yesterday=get_trade_days(end_date=today,count=2)[0]
    trade_date_list=get_trade_dates(20181228,g_count,hold_days)   # 将用于计算的时间序列
    # trade_date_list=dateTransform(trade_date_list)

    # Step 1: 初始化准备数据  PrepareData
    univ_dict,return_df,all_return_df,all_factor_dict,filter_factor_dict=prepareData(trade_date_list,g_factor_list, g_count, filter_factor_list)

    # Step 2: 修理数据
    all_factor_dict, filter_factor_dict=TrimData(univ_dict,all_factor_dict ,filter_factor_dict)

    Package=[univ_dict,return_df,all_return_df,all_factor_dict,filter_factor_dict]

    #使用pickle模块将数据对象保存到文件
    with open('../data/Z1Package_' + file_suffix + '.pkl', 'wb') as pkl_file:
        pickle.dump(Package, pkl_file, 0)