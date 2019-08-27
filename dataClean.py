import datetime
import pandas as pd
import numpy as np
from data import * 
from factor import *
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as st
import scipy.stats as st
import pickle

class dataClean(object):
    def __init__(self, path_minute, path_day):
        self.path_minute = path_minute
        self.path_day = path_day
    
    def filter_factor_rank(self, factor_rank):
        ind_dict = get_ind_dict()
        def series(s):
        #     s = s.sort_values()
            for ind in ind_dict:
                stocks = ind_dict[ind]
                c = s.loc[stocks]
                c_np = c.sort_values().values
                
                c_np = c_np[~np.isnan(c_np)]
                c_max = c_np[-int(len(c_np)*0.1)]
                c_min = c_np[int(len(c_np)*0.1)]
                c = c.map(lambda x: x+4000 if x>=c_max else (x-4000 if x<c_min else x))

                s.loc[stocks] = c
            return s
        return factor_rank.apply(lambda row: series(row), axis=1)
        