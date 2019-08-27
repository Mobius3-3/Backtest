import pandas as pd

class Factor(object):
    def __init__(self, name, max_window, dependencies):
        self.name = name
        self.max_window = max_window
        self.dependencies = dependencies

    def calc(self):
        pass

def calc_factors(universe, factor_list, date):
    factor_dict = dict({})
    for fac in factor_list:
        factor_dict[fac.name] = fac.calc(universe, date)
    return factor_dict

######################################## 因子对象 ###############################################
class gb(Factor):
    def __init__(self,name = 'gb',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class rf(Factor):
    def __init__(self,name = 'rf',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class knn(Factor):
    def __init__(self,name = 'knn',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class svr(Factor):
    def __init__(self,name = 'svr',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class mlp(Factor):
    def __init__(self,name = 'mlp',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class lr(Factor):
    def __init__(self,name = 'lr',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class varm5am10(Factor):
    def __init__(self,name = 'varm5am10',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class varm10am10(Factor):
    def __init__(self,name = 'varm10am10',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class varm15am10(Factor):
    def __init__(self,name = 'varm15am10',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class varm20am10(Factor):
    def __init__(self,name = 'varm20am10',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class lgb_factor_intraday(Factor):
    def __init__(self,name = 'lgb_factor_intraday',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class lgb_factor_daily(Factor):
    def __init__(self,name = 'lgb_factor_daily',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class lgb_factor_daily_test(Factor):
    def __init__(self,name = 'lgb_factor_daily_test',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class lgb_1519(Factor):
    def __init__(self,name = 'lgb_1519',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alpham5am10(Factor):
    def __init__(self,name = 'alpham5am10',max_window = 1,dependencies = ['alpham5am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alpham10am10(Factor):
    def __init__(self,name = 'alpham10am10',max_window = 1,dependencies = ['alpham10am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alpham15am10(Factor):
    def __init__(self,name = 'alpham15am10',max_window = 1,dependencies = ['alpham15am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alpham20am10(Factor):
    def __init__(self,name = 'alpham20am10',max_window = 1,dependencies = ['alpham20am10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad30_vard30(Factor):
    def __init__(self,name = 'alphad30_vard30',max_window = 1,dependencies = ['vard1']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data

class vard1(Factor):
    def __init__(self,name = 'vard1',max_window = 1,dependencies = ['vard1']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data

class vard2(Factor):
    def __init__(self,name = 'vard2',max_window = 1,dependencies = ['vard1']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data

class vard3(Factor):
    def __init__(self,name = 'vard3',max_window = 1,dependencies = ['vard3']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard4(Factor):
    def __init__(self,name = 'vard4',max_window = 1,dependencies = ['vard4']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard5(Factor):
    def __init__(self,name = 'vard5',max_window = 1,dependencies = ['vard5']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard10(Factor):
    def __init__(self,name = 'vard10',max_window = 1,dependencies = ['vard10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard20(Factor):
    def __init__(self,name = 'vard20',max_window = 1,dependencies = ['vard20']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard30(Factor):
    def __init__(self,name = 'vard30',max_window = 1,dependencies = ['vard30']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard60(Factor):
    def __init__(self,name = 'vard60',max_window = 1,dependencies = ['vard60']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard120(Factor):
    def __init__(self,name = 'vard120',max_window = 1,dependencies = ['alphad120']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class vard240(Factor):
    def __init__(self,name = 'vard240',max_window = 1,dependencies = ['alphad240']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad1(Factor):
    def __init__(self,name = 'alphad1',max_window = 1,dependencies = ['alphad1']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data

class alphad2(Factor):
    def __init__(self,name = 'alphad2',max_window = 1,dependencies = ['alphad2']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad3(Factor):
    def __init__(self,name = 'alphad3',max_window = 1,dependencies = ['alphad3']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad4(Factor):
    def __init__(self,name = 'alphad4',max_window = 1,dependencies = ['alphad4']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad5(Factor):
    def __init__(self,name = 'alphad5',max_window = 1,dependencies = ['alphad5']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad10(Factor):
    def __init__(self,name = 'alphad10',max_window = 1,dependencies = ['alphad10']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad20(Factor):
    def __init__(self,name = 'alphad20',max_window = 1,dependencies = ['alphad20']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad30(Factor):
    def __init__(self,name = 'alphad30',max_window = 1,dependencies = ['alphad30']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad60(Factor):
    def __init__(self,name = 'alphad60',max_window = 1,dependencies = ['alphad60']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad120(Factor):
    def __init__(self,name = 'alphad120',max_window = 1,dependencies = ['alphad120']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphad240(Factor):
    def __init__(self,name = 'alphad240',max_window = 1,dependencies = ['alphad240']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class alphard240_md5(Factor):
    def __init__(self,name = 'alphard240_md5',max_window = 1,dependencies = ['alphad240']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self,):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        return data

class pctd1(Factor):
    def __init__(self,name = 'pctd1',max_window = 1,dependencies = ['pctd1']):
        Factor.__init__(self,name,max_window,dependencies)
    
    def calc(self):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data

class pctd60(Factor):
    def __init__(self,name = 'pctd60',max_window = 1,dependencies = ['pctd60']):
        Factor.__init__(self,name,max_window,dependencies)

    def calc(self):
        data = pd.read_pickle('../data/factors/'+self.name+'.pkl')
        # ans = data.loc[date, universe]
        return data
    