import datetime
import pandas as pd
import numpy as np
from data import * 
from factor import Factor,calc_factors
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as st
import scipy.stats as st
import os, time, datetime

class momentum(object):
    def __init__(self, trend: str, dnum: int, dmax: int=None,
                dmin: int=None, mnum: int=None, mmax: int=None,
                mmin: int=None,):
        self.trend = trend #动量或者反转
        self.dnum = dnum
        self.dmax = dmax
        self.dmin = dmin
        self.mnum = mnum
        self.mmax = mmax
        self.mmin = mmin 
    
    def generate(self):
        pass

class dailyMomentum(momentum):
    '''
        dev@计算动量反转因子
        params@trend:'pos' -> momentum or 'neg' -> reverse
        params@dum: [1,2,3,4,5,10,20,30,60,120,240]
        params@dmax: top threshold -> 动量时为动量上限， 反转时为0
        params@dmin: bot threshold -> 动量时为0，反转时为反转下限
    '''
    def __init__(self, trend: str, dum: int, dmax: int=0, dmin: int=0):
        momentum.__init__(self, trend, dum, dmax, dmin)

    # def calc_factors(self):
    #     if self.trend = 'pos':

    def calc_daily_factors(self):
        factor = 'alphad{}'.format(self.dnum)
        factor_df = pd.DataFrame()
        cnt=0
        # print(cnt)
        for root, dirs, files in os.walk('../data/pre_day_data/'):
            for f in files:
                df = pd.read_pickle('../data/pre_day_data/'+f).loc[:,['stockcode',factor]]
                if self.trend == 'pos':
                    if self.dmax>0:
                        df[factor] = df[factor].apply(lambda x: -10 if x>self.dmax else x).rank()
                    else:
                        df[factor] = df[factor].rank()
                elif self.trend == 'neg':
                    if self.dmin<0:
                        df[factor] = df[factor].apply(lambda x: 10 if x<self.dmin else x).rank(ascending=False)
                    else:
                        df[factor] = df[factor].rank(ascending=False)               
                factor_df = factor_df.append(df)
                if cnt%500==0:
                    print(cnt)
                cnt+=1
        factor_df.index = factor_df.index.set_names(['date'])    
        factor_df = factor_df.reset_index().pivot(index='date',columns='stockcode',values=factor)
        return factor_df

class minuteMomentum(momentum):
    '''
        dev@计算动量反转因子
        params@trend:'pos' -> momentum or 'neg' -> reverse
        params@dum: [1,2,3,4,5,10,20,30,60,120,240]
        params@dmax: top threshold -> 动量时为动量上限， 反转时为0
        params@dmin: bot threshold -> 动量时为0，反转时为反转下限
    '''
    def __init__(self, trend: str, dum: int, dmax: int=0, dmin: int=0):
        momentum.__init__(self, trend, dum, dmax, dmin)

    # def calc_factors(self):
    #     if self.trend = 'pos':

    def calc_daily_factors(self):
        factor = 'alphad{}'.format(self.dnum)
        factor_df = pd.DataFrame()
        for root, dirs, files in os.walk('../data/pre_day_data/'):
            for f in files:
                df = pd.read_pickle('../data/pre_day_data/'+f).loc[:,['stockcode',factor]]
                if self.trend == 'pos':
                    if self.dmax>0:
                        df[factor] = df[factor].apply(lambda x: -10 if x>self.dmax else x).rank()
                    else:
                        df[factor] = df[factor].rank()
                elif self.trend == 'neg':
                    if self.dmin<0:
                        df[factor] = df[factor].apply(lambda x: 10 if x<self.dmin else x).rank(ascending=False)
                    else:
                        df[factor] = df[factor].rank(ascending=False)               
                factor_df = factor_df.append(df)
        df.index = df.index.set_names(['date'])    
        factor_df = factor_df.reset_index().pivot(index='date',columns='stockcode',values=factor)
        return factor_df  