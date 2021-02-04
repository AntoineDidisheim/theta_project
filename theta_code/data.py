import pandas as pd
import numpy as np
from parameters import *
import os
import statsmodels.api as sm
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

par = Params()


class RawData:
    def __init__(self, par, freq='monthly'):
        crsp = pd.read_csv(par.data.dir + f'crsp_{freq}.csv')
        # crsp = pd.read_csv(par.data.dir+'crsp_monthly.csv')
        crsp['date'] = pd.to_datetime(crsp['date'], format='%Y%m%d')
        crsp = crsp.rename(columns={'PRC': 'price', 'RET': 'ret', 'VOL': 'vol', 'SHROUT': 'share', 'TICKER': 'tic', 'CFACPR':'adj'}).drop(columns='PERMNO')
        crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
        crsp['tic'] = crsp['tic'].astype('str')
        crsp = crsp.dropna()
        if freq == 'monthly':
            crsp['crsp_date'] = crsp['date'].values
            crsp['date'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
            crsp['date'] = pd.to_datetime(crsp['date'], format='%Y%m')
        self.crsp = crsp

        compustat = pd.read_csv(par.data.dir + 'compustat.csv')
        compustat = compustat.rename(columns={'datadate': 'date', 'fyearq': 'year', 'fqtr': 'quarter', 'atq': 'asset',
                                              'dlttq': 'debt', 'ibq': 'income', 'mkvaltk': 'mkt_cap'}) \
            .drop(columns=['gvkey', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat'])
        compustat['date'] = pd.to_datetime(compustat['date'], format='%Y%m%d')
        compustat['tic'] = compustat['tic'].astype('str')
        if freq == 'monthly':
            compustat['date'] = compustat['date'].dt.year * 100 + compustat['date'].dt.month
            compustat['date'] = pd.to_datetime(compustat['date'], format='%Y%m')
        self.compustat = compustat

        compustat_yearly = pd.read_csv(par.data.dir + 'computstat_yearly.csv')
        compustat_yearly = compustat_yearly.rename(columns={'datadate': 'date', 'fyearq': 'year', 'fqtr': 'quarter', 'ib': 'income', 'ticker': 'tic'
            , 'fyear': 'year'}) \
            .drop(columns=['gvkey', 'indfmt', 'consol', 'date', 'popsrc', 'curcd', 'datafmt', 'costat'])
        compustat_yearly['tic'] = compustat_yearly['tic'].astype('str')
        compustat_yearly = compustat_yearly.dropna()
        compustat_yearly['year'] = compustat_yearly['year'].astype(int)
        self.compustat_yearly = compustat_yearly

        ff = pd.read_csv(par.data.dir + f'ff3_{freq}.csv').merge(pd.read_csv(par.data.dir + f'ffM_{freq}.csv'))
        ff = ff.rename(columns={'Unnamed: 0': 'date', 'Mkt-RF': 'mkt-rf', 'RF': 'rf', 'SMB': 'smb', 'HML': 'hml', 'Mom   ': 'mom'})
        ff.iloc[:, 1:] /= 100
        # crsp = pd.read_csv(par.data.dir+'crsp_monthly.csv')
        if freq == 'daily':
            ff['date'] = pd.to_datetime(ff['date'], format='%Y%m%d')
        else:
            # ff['date'] = pd.to_datetime(ff['date'], format='%Y%m')+pd.DateOffset(months=1)-pd.DateOffset(days=1)
            ff['date'] = pd.to_datetime(ff['date'], format='%Y%m')
        self.ff = ff


class Data:

    def __init__(self, par: Params):
        self.par = par

    def load_and_merge(self):
        opt = self.load_opt()
        pred = self.load_pred()
        opt.head()


        pred.head()
        pred['month'] = pred['crsp_date'].dt.year*100 + pred['crsp_date'].dt.month

        df=pred.merge(opt)

        ind = df[['opt_date', 'tic', 'cp', 'strike']].duplicated()
        df = df.loc[~ind,:]

        # check that we still have the right min number of options
        # df.loc[df['cp'] == 'C', :].groupby(['opt_date', 'tic'])['strike'].count().min()

        ind = df['cp'] == 'C'
        mc = df.loc[ind, :].groupby(['opt_date', 'tic'])['ret'].count().max()
        ind = df['cp'] == 'P'
        mp = df.loc[ind, :].groupby(['opt_date', 'tic'])['ret'].count().max()
        print('Max nb call/put in sample:',mc,mp)

        df['opt_price'] = (df['best_bid']+df['ask'])/2
        ##################
        # process a day
        ##################
        # select a day
        id = df[['opt_date','tic']].iloc[0,:]
        ind = (df['opt_date'] == id['opt_date']) & (df['tic'] == id['tic'])
        day = df.loc[ind,:]

        # create the filler function
        def fill_m_opt(x):
            return np.concatenate([x,np.zeros(Constant.MAX_OPT-len(x))])

        c_ind = day['cp']=='C'
        p_ind = day['cp']=='P'
        kc = fill_m_opt(day.loc[c_ind,'strike'].values)
        kp = fill_m_opt(day.loc[p_ind,'strike'].values)
        calls = fill_m_opt(day.loc[c_ind,'opt_price'].values)
        puts = fill_m_opt(day.loc[p_ind,'opt_price'].values)
        #TODO fix the rf with the proper rf from optionmetrics
        rf = np.array(day.loc[:, 'rf'].iloc[0]).reshape(1)
        fr = np.array(day.loc[:, 'forward_price'].iloc[0]).reshape(1)
        Nc = np.array(c_ind.sum()).reshape(1)
        Np = np.array(p_ind.sum()).reshape(1)
        m = np.concatenate([kc,kp,calls,puts,rf,fr,Nc,Np])


        kc.shape
        calls.shape
        rf.shape

        return df



    def clean_opt_1(self):
        df = pd.read_csv(self.par.data.dir + 'opt.csv')
        del df['issuer']
        del df['exercise_style']
        del df['index_flag']
        # del df['index_flag']
        df['cp_flag'] = df['cp_flag'].astype('category')
        df['ticker'] = df['ticker'].astype('category')
        print(df.head())
        print(df.dtypes)
        df.to_pickle(self.par.data.dir + 'opt_c1.p')


    def clean_opt_2(self):
        df = pd.read_pickle(self.par.data.dir + 'opt_c1.p')
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        df['day'] = df['date'].dt.dayofweek
        df['day'].hist(bins=25)
        plt.title('Day of week no clean')
        plt.savefig('res/data/day_week_split_c1.png')
        plt.close()

        df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
        df['T'] = (df['exdate'] - df['date']).dt.days

        ##################
        # keep only some maturities
        ##################

        # df = df.loc[(df['T'] >= 28) & (df['T'] <= 31), :]
        df = df.loc[(df['T'] == 28), :]

        df['day'] = df['date'].dt.dayofweek
        df['day'].hist(bins=25)
        plt.title('Day of week keep only T=28')
        plt.savefig('res/data/day_week_split_c1_t_clean.png')
        plt.close()

        ##################
        # strike and rename
        ##################
        df = df.rename(columns={'cp_flag':'cp','strike_price':'strike','brest_bid':'bid','best_offer':'ask','ticker':'tic'})
        df['strike'] /= 1000

        ##################
        # Drop nan
        ##################
        df = df.dropna()
        df.to_pickle(self.par.data.dir + 'opt_c2.p')


    def clean_opt_3(self):
        df = pd.read_pickle(self.par.data.dir + 'opt_c2.p')

        ##################
        # remove options with same type, maturity, and strike
        ##################
        ind=df[['date','exdate','tic','cp','strike']].duplicated()
        df=df.loc[~ind,:]


        #################
        # remove  if less than 15 different option strike price
        ##################

        # days with rule --> at least 84.5 different options per maturity
        t=df.groupby(['date','tic'])['strike'].nunique()
        df = df.merge(t.reset_index().rename(columns={'strike':'nb_distinct'}))
        ind=df['nb_distinct']>=15
        print('Drop because less than 5 strike', 1-ind.mean())
        df = df.loc[ind,:]
        del df['nb_distinct']


        ##################
        # remove if volume is too low
        ##################

        # days with rule --> at least 84.5 different options per maturity
        t = df.groupby(['date', 'tic'])['volume'].sum()
        df = df.merge(t.reset_index().rename(columns={'volume': 'vol_tot'}))
        ind = df['vol_tot'] >= 20
        print('Drop volume total below 20', 1 - ind.mean())
        df = df.loc[ind, :]
        del df['vol_tot']


        ##################
        # merge with price for cleaning
        ##################
        raw = RawData(par = self.par, freq='daily')
        raw.crsp.head()
        df.head()
        df = df.merge(raw.crsp[['date','tic','price']])

        ##################
        # remove strike price ratio
        ##################

        df.head()
        df['sp_ratio'] = df['strike']/df['price']
        1-0.2*np.sqrt(1/12)
        t = df.groupby(['tic','date'])['sp_ratio'].agg(['min','max'])
        df = df.merge(t.reset_index())
        ind=(df['min']<1-0.2*np.sqrt(1/12)) & (df['max']>1+0.2*np.sqrt(1/12))
        print('Drop because strike price ratio', 1-ind.mean())
        df = df.loc[ind,:]

        del df['min'], df['max'], df['sp_ratio']
        ##################
        # remove day option when maximum diff between neighbooring strike price is greater than max of 20% o the closing stock price
        ##################
        df=df.sort_values(['date','tic','cp','strike'])
        df.head()
        t=df.groupby(['date','tic','cp'])['strike'].diff()
        df['d'] = t.values
        t=df.groupby(['date','tic'])['d'].max()

        df = df.merge(t.reset_index().rename(columns={'d': 'd_max'}))
        ind=(df['d_max']<= df['price']*0.2) | (df['d_max']<=10)
        print('Drop adjascent price are too close', 1-ind.mean())
        df = df.loc[ind,:]

        df.to_pickle(self.par.data.dir + 'opt_c3.p')

    def clean_opt_4(self):
        df = pd.read_pickle(self.par.data.dir + 'opt_c3.p')
        del df['d']
        del df['d_max']
        df['month'] = df['date'].dt.year*100 + df['date'].dt.month
        df['day'] = df['date'].dt.dayofweek

        # keep only friday to friday
        df = df.loc[df['day']==4,:]

        del df['day']

        raw = RawData(self.par,'daily')
        raw.crsp
        df = df.merge(raw.crsp[['date', 'tic', 'price','adj']].rename(columns={'price': 'S0','adj':'adj0'}))
        df = df.merge(raw.crsp[['date', 'tic', 'price','adj']].rename(columns={'price': 'S28','adj':'adj28','date':'exdate'}))
        df['S28'] = df['S28']* df['adj28']/df['adj0']
        df['ret'] = (df['S28']/df['S0'])-1
        del df['S28'], df['adj0'],df['adj28'],df['optionid'], df['T'], df['volume'], df['price']
        df = df.rename(columns={'date':'opt_date'})
        df.to_pickle(self.par.data.dir + 'opt_final.p')



    def load_opt(self):
        df = pd.read_pickle(self.par.data.dir + 'opt_final.p')

        return df

    def load_pred(self, reload=False):
        if reload:
            ##################
            # recreating the appendix C variable of the Kadan Tang paper (RFS)
            ##################
            var_list = ['mkt_cap', 'btm', 'mom_f', 'inv', 'prof']
            ################# The monthly ones
            raw = RawData(self.par, freq='monthly')
            df = raw.crsp
            ### mkt cap
            df['mkt_cap'] = np.log(1000 * df['share'] * df['price'])
            ### Book to market
            df['year'] = df['date'].dt.year
            comp = raw.compustat.loc[raw.compustat['quarter'] == 4, ['year', 'tic', 'asset', 'debt']].sort_values(['tic', 'year']).reset_index(drop=True)
            comp['asset_l'] = comp.groupby('tic')['asset'].shift(1)
            df = df.merge(comp)
            df['btm'] = 1000 * 1000 * (df['asset'] - df['debt']) / (df['share'] * df['price'])

            ### momentum
            df = df.sort_values(['tic', 'date']).reset_index(drop=True)
            df['r'] = np.log(df['ret'] + 1)
            df.index = df['date']
            t = df.groupby('tic')['r'].rolling(12).sum().reset_index().rename(columns={'r': 'mom_f'})

            df = df.reset_index(drop=True).merge(t)
            ### investment
            df['inv'] = df['asset'] - df['asset_l']
            ### Profitability
            comp = raw.compustat.loc[:, ['date', 'tic', 'asset', 'debt']].sort_values(['tic', 'date']).reset_index(drop=True)
            comp['book_equity'] = comp['asset'] - comp['debt']
            comp['book_equity'] = comp.groupby('tic')['book_equity'].shift(1)
            df = df.merge(comp, how='left')
            df.index = df['date']
            t = df.groupby('tic').apply(lambda x: x['book_equity'].fillna(method='ffill')).reset_index().drop_duplicates()
            df = df.reset_index(drop=True).drop(columns='book_equity')
            df = df.merge(t, how='left')

            comp = raw.compustat_yearly.drop_duplicates()
            df = df.merge(comp)
            df['prof'] = df['income'] / df['book_equity']

            final = df[['tic', 'date', 'crsp_date'] + var_list].copy()

            ################ The daily ones
            #### amihud's liquidity group per month
            raw = RawData(self.par, freq='daily')
            df = raw.crsp.copy()
            df['liq_m'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby(['tic', 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby('tic')['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final = final.merge(t)

            #### amihud's liquidity group per year

            print(final.shape)
            df = raw.crsp.copy().sort_values(['tic', 'date'])
            df['liq_y'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby(['tic', 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby('tic')['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')

            var_list.append('liq_m')
            var_list.append('liq_y')

            ##################
            # adding the one got through regression
            ##################
            df = pd.read_pickle(self.par.data.dir + 'beta_m.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + 'beta_daily.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + 'id_risk_m.p')
            final = final.merge(df)
            # check this one
            df = pd.read_pickle(self.par.data.dir + 'id_risk_daily.p')
            final = final.merge(df)

            var_list.append('beta_monthly')
            var_list.append('beta_daily')
            var_list.append('id_risk_monthly')
            var_list.append('id_risk_daily')

            ##################
            # add macro factor
            ##################
            raw = RawData(par=self.par, freq='monthly')
            final = final.merge(raw.ff)
            var_list + list(raw.ff.iloc[:, 1:].columns)
            final.to_pickle(self.par.data.dir + 'pred.p')
        else:
            final = pd.read_pickle(self.par.data.dir + 'pred.p')
        return final

    @staticmethod
    def get_beta(freq='monthly'):
        print('#' * 50)
        print('Start Beta', freq)
        print('#' * 50)
        if freq == 'monthly':
            r = 60
        else:
            r = 12

        raw = RawData(self.par, freq)
        df = raw.ff.merge(raw.crsp, how='inner', on='date')
        df['one'] = 1

        def get_delta(x):
            if x.shape[0] > r / 2:
                return sm.OLS(x['ret'] - x['rf'], x[['one', 'mkt-rf']]).fit().params['mkt-rf']
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('tic').apply(get_delta)
            t.name = date
            res.append(t)
            date += pd.DateOffset(months=1)
            date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.month == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'tic', 'value': f'beta_{freq}'})
        return t

    @staticmethod
    def get_id_risk(freq='monthly'):
        print('#' * 50)
        print('Start id risk', freq)
        print('#' * 50)
        if freq == 'monthly':
            r = 60
        else:
            r = 12

        raw = RawData(par, freq)
        df = raw.ff.merge(raw.crsp, how='inner', on='date')
        df['one'] = 1
        df = df.dropna()

        def get_std(x):
            if x.shape[0] > r / 2:
                return np.std(sm.OLS(x['ret'] - x['rf'], x[['one', 'mkt-rf']]).fit().resid)
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')

        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('tic').apply(get_std)
            t.name = date
            res.append(t)
            date += pd.DateOffset(months=1)
            date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.month == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'tic', 'value': f'id_risk_{freq}'})
        return t

    def gen_delta(self):
        df = Data.get_beta('monthly')
        df.to_pickle(self.par.data.dir + 'beta_m.p')
        df = Data.get_beta('daily')
        df.to_pickle(self.par.data.dir + 'beta_daily.p')

    def gen_id_risk(self):
        df = Data.get_id_risk('monthly')
        df.to_pickle(self.par.data.dir + 'id_risk_m.p')
        df = Data.get_id_risk('daily')
        df.to_pickle(self.par.data.dir + 'id_risk_daily.p')

    def gen_all(self):
        self.get_beta()
        self.gen_id_risk()


self = Data(Params())
# self.clean_opt_3()
# self.load_pred()
# self.gen_delta()
# self.gen_id_risk()
