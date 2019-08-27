import datetime
import time,os
from data import *
from multiprocessing.dummy import Pool as ThreadPool
from factor import Factor,calc_factors
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import pickle
import matplotlib.pyplot as plt
from pylab import mpl
import json 

def get_univ_by_factor(factor, univ_dict, num = 500, ascend=False,threshold=0):
    '''
        dev@根据日间因子选择当个交易日的股票池
        params@univ_dict: 根据选股条件筛选的股票池
        params@ind_rate: 逐日中证500各行业占比
        params@ind_dict: 各行业的股票池
        return@逐日的500支符合要求的股票
    '''
    ind_rate = get_ind_rate()
    ind_dict = get_ind_dict()
    univ_factor_dict = {}
    for date in univ_dict.keys():
        univ_factor_dict[date] = []
        for ind in ind_dict.keys():
            num_stock = round(num * ind_rate.loc[date, ind])
            # print('num_stock',num_stock)
            stocks = ind_dict[ind]
            if threshold == 0:
                univ_factor_dict[date] += factor.loc[date,stocks].sort_values(ascending=ascend).index[:int(num_stock)].tolist()
            else:
                stocks = factor.loc[date,stocks].sort_values(ascending=ascend).index.tolist()
                stocks = stocks[int(0.2*len(stocks)):]
                univ_factor_dict[date] += stocks[:int(num_stock)]
    return univ_factor_dict

def get_univ_ind(date, stockpool, num=150):
    ind_rate = get_ind_rate()
    ind_dict = get_ind_dict()
    univ_ind = []
    for ind in ind_dict:
        num_stock = int(num * ind_rate.loc[date, ind])
        stocks = ind_dict[ind]
        univ_ind += [s for s in stockpool if s in stocks][:num_stock]
    # if len(univ_ind)>100:
    #     univ_ind = univ_ind[:100]
    return univ_ind


def neutralize_Return_calculator(factor, univ_factor_dict, all_return_df):
    '''
        dev@根据日内因子值及行业中性选择当日的股票
        params@factor: 日内因子值
        params@univ_dict: 根据选股条件以及日间条件筛选的股票池
        params@all_return: 从10点开始买入至换仓的收益
        params@ind_rate: 逐日中证500各行业占比
        params@ind_dict: 各行业的股票池    
        return@逐日的日间日内组合因子选出top100支股票的收益
    '''
    univ_dict = get_univ_by_factor(factor, univ_factor_dict, num=100, ascend=False, threshold =0)
    
    all_date_list = list(all_return_df.index)
    date_list = list(univ_dict.keys())
    Ret_df = pd.DataFrame(index=all_date_list, columns=[0])

    for n in range(len(date_list)-1):
        start = date_list[n]
        end = date_list[n+1]
        group_stock_day_minute = univ_dict[start]

        return_openam10 = get_return_openam10().loc[start,group_stock_day_minute].values
        # print(return_openam10)
        # 计算持仓期的累计收益率，并加入千2的滑点
        if all_date_list.index(start) == 0:
            start_before = start-1
        else:
            start_before = all_date_list[all_date_list.index(start)-1]
        end_before = all_date_list[all_date_list.index(end)-1]
        # print(all_date_list,all_date_list.index(start),all_date_list.index(end))
        cumret=all_return_df.loc[start_before:end,group_stock_day_minute].shift(1)+1
        # print(cumret.index)
        cumret = cumret.loc[start:end,:]
        cumret.loc[start,:] = 1+return_openam10

        # print(cumret.index)
        cumret = cumret.cumprod().mean(axis=1)
        cumret.loc[end] *= 1-2e-3
        start_value = cumret.loc[start]-1
        Ret_df.loc[start:end,0]=cumret.pct_change()
        Ret_df.loc[start,0] =  start_value
    return Ret_df

def Return_multi_factor(factor_dict, all_return_df, filter_factor_dict, univ_dict, is_filter=False):
    '''
        dev@根据日间日内因子的组合计算逐日收益率
        params@factor_dict: key分别为'day' -> 日间因子, 'minute' -> 日内因子
    '''
    all_date_list = list(all_return_df.index)
    date_list = list(univ_dict.keys())
    Ret_df = pd.DataFrame(index=all_date_list, columns=[0])

    for n in range(len(date_list)-1):
        start = date_list[n] # 起始date
        end = date_list[n+1]
        univ = univ_dict[start]
        univ = set(univ) & set(factor_dict['day'].loc[start].dropna().index) & set(factor_dict['minute'].loc[start].dropna().index)
        factor_day_stock = list(factor_dict['day'].loc[start,univ].sort_values(ascending=False).index)
        factor_minute_stock_temp = list(factor_dict['minute'].loc[start,univ].sort_values(ascending=False).index)
        len_fm = len(factor_minute_stock_temp)
        factor_minute_stock = factor_minute_stock_temp[int(len_fm*0.2):-int(len_fm*0.2)]
        # 日间最小的30%， 日内最大20%-40%, 按日间优先进行选股
        group_stock_day = factor_day_stock[:int(0.6*len(factor_day_stock))]
        # group_stock_day = factor_day_stock[:int(0.3*len(factor_day_stock))]
        group_stock_minute = factor_minute_stock
        # group_stock_minute = factor_minute_stock[int(0.2*len(factor_day_stock)):int(0.4*len(factor_day_stock))]

        # group_stock_day_minute = []
        # for s in group_stock_day:
        #     if s in group_stock_minute and len(group_stock_day_minute) < 100:
        #         group_stock_day_minute.append(s)

        group_stock_day_minute = []
        for s in group_stock_day:
            if s in group_stock_minute:
                group_stock_day_minute.append(s)

        group_stock_day_minute = get_univ_ind(start, group_stock_day_minute, num=120)
        print(len(group_stock_day_minute))

        return_openam10 = get_return_openam10().loc[start,group_stock_day_minute].values
        # print(return_openam10)
        # 计算持仓期的累计收益率，并加入千2的滑点
        if all_date_list.index(start) == 0:
            start_before = start-1
        else:
            start_before = all_date_list[all_date_list.index(start)-1]
        end_before = all_date_list[all_date_list.index(end)-1]
        # print(all_date_list,all_date_list.index(start),all_date_list.index(end))
        cumret=all_return_df.loc[start_before:end,group_stock_day_minute].shift(1)+1
        # print(cumret.index)
        cumret = cumret.loc[start:end,:]
        cumret.loc[start,:] = 1+return_openam10

        # print(cumret.index)
        cumret = cumret.cumprod().mean(axis=1)
        cumret.loc[end] *= 1-2e-3
        start_value = cumret.loc[start]-1
        Ret_df.loc[start:end,0]=cumret.pct_change()
        if n ==0:
            Ret_df.loc[start,0] =  0
        else:
            Ret_df.loc[start,0] =  start_value
        print(start_value)
        # print(Ret_df)
    return Ret_df

def all_Group_Return_calculator(factor, univ_dict, all_return_df, filter_factor_dict, GroupNum=10, is_filter=False, factor_type='minute'):
    all_date_list = list(all_return_df.index)
    # print(all_date_list)
    date_list = list(univ_dict.keys())
    all_Group_Ret_df = pd.DataFrame(index=all_date_list, columns=\
                                    list(np.array(range(GroupNum))))
    for n in range(len(date_list)-1):
        # 第n天date
        start = date_list[n]
        # 第n+hold_days天date
        end = date_list[n+1]
        # 第n天股票池
        univ=univ_dict[start]
        # 第n天股票池中删除缺失因子值的股票
        univ=set(univ)&set(factor.loc[start].dropna().index)
        # 根据第n天因子值递增排序的股票池
        factor_se_stock=list(factor.loc[start,univ].dropna().sort_values().index)

        # 剔除filter_factor_dict中后10%的股票
        if is_filter:
            filter_list = []
            for fac in filter_factor_dict:
                filter_stocks = filter_factor_dict[fac].loc[start, factor_se_stock].sort_values().index.tolist()
                filter_list += filter_stocks[:int(0.1*len(filter_stocks))]
            factor_se_stock = [s for s in factor_se_stock if s not in list(set(filter_list))]

        N=len(factor_se_stock)
        for i in range(GroupNum):
            if factor_type == 'minute':
                group_stock=factor_se_stock[int(N/GroupNum*i):int(N/GroupNum*(i+1))]
                return_openam10 = get_return_openam10().loc[start,group_stock].values
                # 计算持仓期的累计收益率，并加入千2的滑点
                if all_date_list.index(start) == 0:
                    start_before = start-1
                else:
                    start_before = all_date_list[all_date_list.index(start)-1]
                end_before = all_date_list[all_date_list.index(end)-1]
                # print(all_date_list,all_date_list.index(start),all_date_list.index(end))
                cumret=all_return_df.loc[start_before:end,group_stock].shift(1)+1
                # print(cumret.index)
                cumret = cumret.loc[start:end,:]
                cumret.loc[start,:] = 1+return_openam10
                # print(cumret.index)
                cumret = cumret.cumprod().mean(axis=1).shift(1)
                cumret.loc[end] *= 1-2e-3
                all_Group_Ret_df.loc[start:end,i]=cumret.pct_change().shift(-1)                
                
            else:
                group_stock=factor_se_stock[int(N/GroupNum*i):int(N/GroupNum*(i+1))]
                # 计算持仓期的累计收益率，并加入千2的滑点
                cumret=(all_return_df.loc[start:end,group_stock]+1).cumprod().mean(axis=1).shift(1).fillna(1)
                cumret.loc[end] *= 1-2e-3
                all_Group_Ret_df.loc[start:end,i]=cumret.pct_change().shift(-1)
                #(((all_return_df.loc[start:end,group_stock]+1).cumprod()-1).mean(axis=1)+1).pct_change().shift(-1)
    old_index = all_Group_Ret_df.index.tolist()
    new_index = old_index[old_index.index(date_list[0]):]
    all_Group_Ret_df=all_Group_Ret_df.loc[new_index].shift(1).fillna(0)
    return all_Group_Ret_df       

def Group_Return_calculator(all_Group_Ret_df,univ_dict,GroupNum=10):
    GroupRet_df=pd.DataFrame(index=list(list(univ_dict.keys())),columns=list(np.array(range(GroupNum))))
    univ_dict_list = list(univ_dict.keys())
    all_dates = all_Group_Ret_df.index.tolist()

    for date in univ_dict_list:    #这个也是个循环
        start = date
        if date == univ_dict_list[-1]:
            end_before = all_dates[-1]
        else:
            end = univ_dict_list[univ_dict_list.index(start)+1]
            end_before = all_dates[all_dates.index(end)-1]
       
        GroupRet_df.loc[date, :] = (all_Group_Ret_df.loc[start:end_before,:]+1).cumprod().loc[end_before] - 1

    return GroupRet_df

def ic_calculator(factor,return_df,univ_dict):
    ic_list=[]
    p_value_list=[]
    for date in list(univ_dict.keys()):   #这里是循环
        univ=univ_dict[date]
        univ=list(set(univ)&set(factor.loc[date].dropna().index)&set(return_df.loc[date].dropna().index))
        if len(univ)<10:
            continue
        factor_se=factor.loc[date,univ]
        return_se=return_df.loc[date,univ]
        ic,p_value=st.spearmanr(factor_se,return_se)
        ic_list.append(ic)
        p_value_list.append(p_value)
    return ic_list

# 获取指数收益
def get_index_return(univ_dict,index=500,count=980):
    trade_date_list=list(univ_dict.keys())
    date=max(trade_date_list)
    price=get_price(index,end_date=date,count=count,fields=['close'])['500closed']
    price_index = price.index.tolist()
    # print(price_index)
    # print(trade_date_list[0],price_index.index(trade_date_list[0]))
    new_index = price_index[price_index.index(trade_date_list[0]):price_index.index(trade_date_list[-1])+1]
    price_return=price.loc[new_index].pct_change().fillna(0)
    price_return_by_tradeday=price.loc[trade_date_list].pct_change().fillna(0)
    return price_return,price_return_by_tradeday

# 
def effect_test(univ_dict,key,group_return,group_excess_return,index_return):

    y,m,d=time.strptime(str(list(univ_dict.keys())[-1]),'%Y%m%d')[:3]
    e = datetime.date(y,m,d)
    y,m,d=time.strptime(str(list(univ_dict.keys())[0]),'%Y%m%d')[:3]
    s = datetime.date(y,m,d)
    daylength=(e-s).days

    # 计算策略年化收益和指数年化收益
    annual_return=np.power(np.cumprod(group_return+1).iloc[-1,:],365/daylength).astype('float64')
    index_annual_return=np.power((index_return+1).cumprod().iloc[-1],365/daylength)

    # Test One: 组合序列与组合收益的相关性，相关性大于0.5
    sequence=pd.Series(np.array(range(10)))
    # 年化收益序列和排序的相关性
    test_one_corr=annual_return.corr(sequence)
    test_one_passgrade=0.5
    test_one_pass=abs(test_one_corr)>test_one_passgrade
    
    if test_one_corr<0:
        wingroup,losegroup=0,9
    else:
        wingroup,losegroup=9,0
        
    # Test Two: 赢家组合明显跑赢市场，输家组合明显跑输市场，程度大于5%     
    test_two_passgrade=0.05
    test_two_win_pass=annual_return[wingroup]-index_annual_return>test_two_passgrade
    test_two_lose_pass=index_annual_return-annual_return[losegroup]>test_two_passgrade
    test_two_pass=test_two_win_pass&test_two_lose_pass

    # Test Tree: 高收益组合跑赢基准的概率，低收益组合跑输基准的概率，概率大小0.5
    test_three_grade=0.5
    test_three_win_pass=(group_excess_return[wingroup]>0).sum()/len(group_excess_return[wingroup])>0.5
    test_three_lose_pass=(group_excess_return[losegroup]<0).sum()/len(group_excess_return[losegroup])>0.5
    test_three_pass=test_three_win_pass&test_three_lose_pass

    return [test_one_pass,test_two_win_pass,test_two_lose_pass,test_three_win_pass,test_three_lose_pass]

#2. 计算绩效
def plot_nav(all_return_df,index_return,key,path):
# Preallocate figures
    fig = plt.figure(figsize=(12,6))
    fig.set_facecolor('white')
    fig.set_tight_layout(True)
    # ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(111)
    # ax1.grid()
    ax2.grid()
    # ax1.set_ylabel(u"净值", fontsize=16)
    ax2.set_ylabel(u"对冲净值", fontsize=16)
    # ax1.set_title(u"因子选股 - 净值走势",fontsize=16)
    ax2.set_title(u"因子选股 - 对冲中证500指数后净值走势", fontsize=16)
# preallocate data    
    date=list(all_return_df.index)
    index_return = index_return.loc[date]
    date = [datetime.date(d//10000,d//100%100,d%100) for d in date]
    # print(date[0])
    sequence=all_return_df.columns
    # nav_dict = {}
    # nav_excess_dict={}
# plot nav
    
    for sq in sequence:
        # nav=(1+all_return_df[sq]).cumprod()
        nav_excess=(1+all_return_df[sq]-index_return).cumprod()
        # ax1.plot(date,nav,label=str(sq))
        ax2.plot(date,nav_excess,label=str(sq))
        # nav_dict[sq] = nav
        # nav_excess_dict[sq] = nav_excess
    # pd.DataFrame(data=nav_dict).to_csv('../data/nav.csv',encoding='gbk')
    # pd.DataFrame(data=nav_excess_dict).to_csv('../data/nav_excess.csv',encoding='gbk')
    # with open('../data/nav.json', 'w') as f1:
    #     json.dump(nav_dict, f1)
    # with open('../data/nav_excess.json', 'w') as f2:
    #     json.dump(nav_excess, f2)  
         
    # ax1.legend(loc=0,fontsize=12)
    ax2.legend(loc=0,fontsize=12)
    plt.savefig(path+'fig/'+key+'.png')
    plt.show()

def plot_excess_nav(all_return_df,index_return,key,path):
# Preallocate figures
    fig = plt.figure(figsize=(12,6))
    fig.set_facecolor('white')
    fig.set_tight_layout(True)
    # ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(111)
    # ax1.grid()
    ax2.grid()
    # ax1.set_ylabel(u"净值", fontsize=16)
    ax2.set_ylabel(u"对冲净值", fontsize=16)
    # ax1.set_title(u"因子选股 - 净值走势",fontsize=16)
    ax2.set_title(u"因子选股 - 对冲中证500指数后净值走势", fontsize=16)
# preallocate data    
    date=list(all_return_df.index)
    index_return = index_return.loc[date]
    date = [datetime.date(d//10000,d//100%100,d%100) for d in date]
    # print(date[0])
    sequence=all_return_df.columns
    # nav_dict = {}
    # nav_excess_dict={}
# plot nav
    
    for sq in sequence:
        # nav=(1+all_return_df[sq]).cumprod()
        nav_excess=(1+all_return_df[sq]).cumprod()
        # ax1.plot(date,nav,label=str(sq))
        ax2.plot(date,nav_excess,label=str(sq))
        # nav_dict[sq] = nav
        # nav_excess_dict[sq] = nav_excess
    # pd.DataFrame(data=nav_dict).to_csv('../data/nav.csv',encoding='gbk')
    # pd.DataFrame(data=nav_excess_dict).to_csv('../data/nav_excess.csv',encoding='gbk')
    # with open('../data/nav.json', 'w') as f1:
    #     json.dump(nav_dict, f1)
    # with open('../data/nav_excess.json', 'w') as f2:
    #     json.dump(nav_excess, f2)  
         
    # ax1.legend(loc=0,fontsize=12)
    ax2.legend(loc=0,fontsize=12)
    plt.savefig(path+'fig/'+key+'.png')
    plt.show()
    
def polish(x):
    return '%.2f%%' % (x*100)

def result_stats(key,all_return_df,index_return):  

    # Preallocate result DataFrame
    sequences=all_return_df.columns

    cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'),
            (u'纯多头', u'年化收益'), (u'纯多头', u'最大回撤'), (u'纯多头', u'收益波动率'), 
            (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
    columns = pd.MultiIndex.from_tuples(cols)
    result_df = pd.DataFrame(index = sequences,columns=columns)
    result_df.index.name = "%s" % (key)

    # index_return = index_return.shift(-1).fillna(0)

    for sq in sequences:  #循环在这里开始

        # 净值
        return_data=all_return_df[sq].fillna(0)

        return_data_excess=return_data-index_return
        nav=(1+return_data).cumprod()
        nav_excess=(1+return_data_excess).cumprod()
        nav_index=(1+index_return).cumprod()

        # Beta
        # print(return_data,index_return)
        beta=return_data.corr(index_return)*return_data.std()/index_return.std()
        beta_excess=return_data_excess.corr(index_return)*return_data_excess.std()/index_return.std()

        #年化收益 datetime.date(d//10000,d//100%100,d%100)
        d1 = return_data.index[-1]
        d2 = return_data.index[0]
        daylength=(datetime.date(d1//10000,d1//100%100,d1%100)-\
                    datetime.date(d2//10000,d2//100%100,d2%100)).days#######
        yearly_return=np.power(nav.iloc[-1],1.0*365/daylength)-1
        yearly_return_excess=np.power(nav_excess.iloc[-1],1.0*365/daylength)-1
        yearly_index_return=np.power(nav_index.iloc[-1],1.0*365/daylength)-1

        # 最大回撤 其实这个完全看不懂
        max_drawdown=max([1-v/max(1,max(nav.iloc[:i+1])) for i,v in enumerate(nav)])
        max_drawdown_excess=max([1-v/max(1,max(nav_excess.iloc[:i+1])) for i,v in enumerate(nav_excess)])

        # 波动率
        vol=return_data.std()*np.sqrt(252)
        vol_excess=return_data_excess.std()*np.sqrt(252)

        # Alpha
        rf=0.04
        alpha=yearly_return-(rf+beta*(yearly_return-yearly_index_return))
        alpha_excess=yearly_return_excess-(rf+beta_excess*(yearly_return-yearly_index_return))

        # 信息比率
        ir=(yearly_return-yearly_index_return)/(return_data_excess.std()*np.sqrt(252))

        # 夏普比率
        sharpe=(yearly_return_excess-rf)/vol_excess

        # 美化打印

        alpha,yearly_return,max_drawdown,vol,yearly_return_excess,max_drawdown_excess,vol_excess=\
        map(polish,[alpha,yearly_return,max_drawdown,vol,yearly_return_excess,max_drawdown_excess,vol_excess])
        sharpe=round(sharpe,2)
        ir=round(ir,2)
        beta=round(ir,2)

        result_df.loc[sq]=[alpha,beta_excess,ir,sharpe,yearly_return,max_drawdown,vol,yearly_return_excess,max_drawdown_excess,vol_excess]
    return result_df

def draw_excess_return(key,excess_return,path):
    excess_return_mean=excess_return[1:].mean()
    excess_return_mean.index = map(lambda x:int(x)+1,excess_return_mean.index)
    excess_plus=excess_return_mean[excess_return_mean>0]
    excess_minus=excess_return_mean[excess_return_mean<0]

    fig = plt.figure(figsize=(12, 6))
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(111)
    ax1.bar(excess_plus.index, excess_plus.values, align='center', color='r', width=0.35)
    ax1.bar(excess_minus.index, excess_minus.values, align='center', color='g', width=0.35)
    ax1.set_xlim(left=0.5, right=len(excess_return_mean)+0.5)
    ax1.set_ylabel(u'超额收益', fontsize=16)
    ax1.set_xlabel(u'十分位分组', fontsize=16)
    ax1.set_xticks(excess_return_mean.index)
    ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontsize=14)
    ax1.set_yticklabels([str(round(x*100,1))+'0%' for x in ax1.get_yticks()], fontsize=14)
    ax1.set_title(u"因子选股分组超额收益", fontsize=16)
    plt.savefig(path+'fig/alphaReturn_'+key+'.png')
    ax1.grid()

# --------------------------------- 执行程序 ----------------------------------

if __name__ == 'main':

    pd.set_option('display.max_rows',None)
    mpl.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  

    with open('../data/Z1Package.pkl', 'rb') as pkl_file:
        load_Package = pickle.load(pkl_file)
    univ_dict,return_df,all_return_df,all_factor_dict,filter_factor_dict=load_Package

    print('计算IC_IR......')
    ic_list_dict={}
    for key,factor in all_factor_dict.items():
        ic_list=ic_calculator(factor,return_df,univ_dict)
        ic_list_dict[key]=ic_list
    # 整理结果
    ic_df=pd.DataFrame(ic_list_dict,index=list(univ_dict.keys())[:-1])
    ic_ir_se=ic_df.mean()/ic_df.std()

    print('计算分组收益......')
    GroupNum=10
    all_Factor_Group_Return_dict={}
    Factor_Group_Return_dict={}
    for key,factor in all_factor_dict.items():
    # 全return    
        all_GroupRet_df=all_Group_Return_calculator(factor,univ_dict,all_return_df,filter_factor_dict,GroupNum,False)
        all_Factor_Group_Return_dict[key]=all_GroupRet_df
    # 调仓期return    
        GroupRet_df=Group_Return_calculator(factor,univ_dict,return_df,GroupNum)   
        Factor_Group_Return_dict[key]=GroupRet_df
        
    print('计算指数收益......')
    index=500
    index_return,index_return_by_tradeday=get_index_return(univ_dict,index,711)
    # Factor_Group_Excess_Return_dict={}
    for key,group_return in Factor_Group_Return_dict.items():
        Factor_Group_Excess_Return_dict[key]=group_return.subtract(index_return_by_tradeday,axis=0)

    print('因子有效性测试......')
    effect_test_dict={}
    for key,group_return in Factor_Group_Return_dict.items():
        group_excess_return=Factor_Group_Excess_Return_dict[key]   
        effect_test_dict[key]=effect_test(univ_dict,key,group_return,group_excess_return,index_return)
        
    #----------有效因子列表-----------
    effect_factor_list=[]
    for key,effect in effect_test_dict.items():
        if all(effect):
            effect_factor_list.append(key)
    effect_factor_list
    #------------有效因子-------------
    effect_factor_dict={key:value for key,value in all_factor_dict.items() if key in effect_factor_list}

    print('完成')

    for key,factor in effect_factor_dict.items():
    # key='pctd1'
        print(key)
        plot_nav(all_Factor_Group_Return_dict[key],index_return,key,path) 
