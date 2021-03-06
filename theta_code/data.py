import pandas as pd
import numpy as np
from parameters import *
import os
import statsmodels.api as sm
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

class RawData:
    def __init__(self, par, freq='monthly'):

        ff = pd.read_csv(par.data.dir + f'raw/ff3_{freq}.csv').merge(pd.read_csv(par.data.dir + f'raw/ffM_{freq}.csv'))
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
        self.m_df = None
        self.p_df = None
        self.label_df = None

        self.train_m_df = None
        self.train_p_df = None
        self.train_label_df = None

        self.test_m_df = None
        self.test_p_df = None
        self.test_label_df = None

        self.ind_order_list = None
        self.shuffle_id = 0

        ##################
        # set name
        ##################
        self.name = 'd_'
        if self.par.data.opt:
            self.name+='Opt'
        if self.par.data.comp:
            self.name+='Comp'
        if self.par.data.crsp:
            self.name+='Crsp'

        self.name+=str(self.par.data.opt_smooth.name)


        ##################
        # make dir
        ##################
        self.make_dir(f"{self.par.data.dir}int_gen")
        self.make_dir(f"{self.par.data.dir}{self.name}/side")
        self.make_dir(f'{self.par.data.dir}{self.name}/int/')



    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)



    def set_year_test(self,year):

        test_id = np.array(list(self.label_df.index[self.label_df['date'].dt.year == year]))
        np.random.shuffle(test_id)
        if self.par.model.cv == CrossValidation.YEAR_BY_YEAR:
            train_id = np.array(list(self.label_df.index[self.label_df['date'].dt.year != year]))
        if self.par.model.cv == CrossValidation.EXPANDING:
            train_id = np.array(list(self.label_df.index[self.label_df['date'].dt.year < year]))
        np.random.shuffle(train_id)


        self.test_label_df = self.label_df.iloc[test_id, :].reset_index(drop=True)
        self.test_m_df = self.m_df.iloc[test_id, :].reset_index(drop=True)
        self.test_p_df = self.p_df.iloc[test_id, :].reset_index(drop=True)

        self.train_label_df = self.label_df.iloc[train_id, :].reset_index(drop=True)
        self.train_m_df = self.m_df.iloc[train_id, :].reset_index(drop=True)
        self.train_p_df = self.p_df.iloc[train_id, :].reset_index(drop=True)

        print(f'Set training year {year}')




    def move_shuffle(self):
        if self.ind_order_list is None:
            t = list(self.label_df.index.copy())
            np.random.shuffle(t)
            self.ind_order_list = np.array(t)
            self.shuffle_id = 0
        p_test = self.par.data.val_split
        t_size = int(np.ceil(len(self.ind_order_list) * p_test))
        start = self.shuffle_id * t_size
        end = min((self.shuffle_id + 1) * t_size, len(self.ind_order_list))
        test_id = self.ind_order_list[start:end]
        train_id = np.array([x for x in self.ind_order_list if x not in test_id])

        self.test_label_df = self.label_df.iloc[test_id, :].reset_index(drop=True)
        self.test_m_df = self.m_df.iloc[test_id, :].reset_index(drop=True)
        self.test_p_df = self.p_df.iloc[test_id, :].reset_index(drop=True)

        self.train_label_df = self.label_df.iloc[train_id, :].reset_index(drop=True)
        self.train_m_df = self.m_df.iloc[train_id, :].reset_index(drop=True)
        self.train_p_df = self.p_df.iloc[train_id, :].reset_index(drop=True)
        self.shuffle_id += 1
        if self.shuffle_id * p_test > 1.0:
            self.shuffle_id = 0
        print(f'Set shuffle {self.shuffle_id}')

    def load_final(self):
        # finally save all
        self.m_df = pd.read_pickle(self.par.data.dir + f'{self.name}/m_df.p').reset_index(drop=True)
        self.p_df = pd.read_pickle(self.par.data.dir + f'{self.name}/pred_df.p').reset_index(drop=True)
        self.label_df = pd.read_pickle(self.par.data.dir + f'{self.name}/label_df.p').reset_index(drop=True)

        # deal with remaining inf
        self.p_df = self.p_df.replace({-np.inf: np.nan, np.inf: np.nan})

        for c in self.p_df.columns:
            self.p_df.loc[:, c] = (self.p_df[c] - self.p_df[c].mean()) / self.p_df[c].std()

        self.p_df = self.p_df.fillna(0)
        # self.p_df.quantile(0.999)
        self.p_df = self.p_df.clip(-3.0, 3.0)

    def load_and_merge_pred_opt(self):
        cleaner = pd.read_pickle(self.par.data.dir + f'{self.name}/side/cleaner_all.p')
        opt = self.load_opt()

        if self.par.data.crsp:
            if self.par.data.comp:
                pred = self.load_pred_compustat_and_crsp()
            else:
                pred = self.load_pred_crsp_only()

        # pred['month'] = pred['date'].dt.year * 100 + pred['date'].dt.month
        # del pred['date']

        df = pred.merge(opt, on=['gvkey', 'date'])
        cleaner = cleaner.append(self.count_sample(df, 'Merge with predictors'))

        df = df.dropna()
        cleaner = cleaner.append(self.count_sample(df, 'Drop missing predictors'))
        print(cleaner)

        ind = df['cp'] == 'C'
        mc = df.loc[ind, :].groupby(['date', 'gvkey'])['ret'].count().max()
        ind = df['cp'] == 'P'
        mp = df.loc[ind, :].groupby(['date', 'gvkey'])['ret'].count().max()
        print('Max nb call/put in sample:', mc, mp)
        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_after_merge.csv')
        return df

    def create_a_dataset(self):

        if self.par.data.crsp:
            if self.par.data.comp:
                pred = self.load_pred_compustat_and_crsp()
            else:
                pred = self.load_pred_crsp_only()

        # specify here the list of DataType which do not need to be merge with any dataset
        if self.par.data.opt and ~self.par.data.comp and ~self.par.data.crsp:
            df = self.load_opt()
            raw = RawData(self.par, 'daily')
            df = df.merge(raw.ff[['date', 'rf']])
            pred_col = []
        else:
            df = self.load_and_merge_pred_opt()
            pred_col = list(pred.drop(columns=['gvkey', 'date']).columns)

        ##################
        # process a day
        ##################
        # select a day
        t = df[['date', 'gvkey', 'ret']].drop_duplicates().reset_index(drop=True)
        M = []
        P = []
        # i = 1000
        for i in range(t.shape[0]):
            id = t[['date', 'gvkey']].iloc[i, :]
            ind = (df['date'] == id['date']) & (df['gvkey'] == id['gvkey'])
            day = df.loc[ind, :]

            m, p = self.pre_process_day(day, pred_col)

            M.append(m)
            P.append(p)
            if i % 100 == 0:
                print(i, '/', t.shape[0])

        iv_col = ['iv' + str(x) for x in np.arange(80, 130, 10)]
        if self.par.data.opt:
            pred_col = pred_col + iv_col
        m_df = pd.DataFrame(M)
        p_df = pd.DataFrame(P, columns=pred_col)
        # m_df = self.m_df
        # p_df = self.p_df
        m_df.dropna()

        # find na in any df
        temp = pd.concat([t, m_df, p_df], axis=1)

        ind = pd.isna(temp).sum(1) == 0

        m_df = m_df.loc[ind, :].reset_index(drop=True)
        p_df = p_df.loc[ind, :].reset_index(drop=True)
        t = t.loc[ind, :].reset_index(drop=True)

        # finally save all
        m_df.to_pickle(self.par.data.dir + f'{self.name}/m_df.p')
        p_df.to_pickle(self.par.data.dir + f'{self.name}/pred_df.p')
        t.to_pickle(self.par.data.dir + f'{self.name}/label_df.p')

        # debug
        # m_df = pd.read_pickle(self.par.data.dir+f'{self.name}/m_df.p')
        # p_df = pd.read_pickle(self.par.data.dir+f'{self.name}/pred_df.p')
        # t = pd.read_pickle(self.par.data.dir+f'{self.name}/label_df.p')

    def pre_process_day(self, day, pred_col):
        # create the filler function
        def fill_m_opt(x):
            return np.concatenate([x, np.zeros(Constant.MAX_OPT - len(x))])

        s0 = np.array(day.loc[:, 'S0'].iloc[0]).reshape(1)
        rf = np.array(day.loc[:, 'rf'].iloc[0]).reshape(1)

        t = day.loc[:, ['strike', 'opt_price']].copy().sort_values(['strike', 'opt_price']).reset_index(drop=True).groupby('strike').mean().reset_index()

        day.loc[day['cp'] == 'P', 'strike'].max()

        # cb = CubicSpline(t['strike'], t['opt_price'])
        bound=(t['opt_price'].iloc[0], t['opt_price'].iloc[-1])
        cb = interp1d(t['strike'],t['opt_price'], bounds_error=False, fill_value=bound)
        if self.par.data.opt_smooth == OptSmooth.EXT:
            K = np.linspace(s0*0.5,s0*1.5,200)
        if self.par.data.opt_smooth == OptSmooth.INT:
            K = np.linspace(t['strike'].min(), t['strike'].max(), 200)
        assert len(K) == 200, 'Problem with the linespace'

        PRICE = cb(K)
        #
        # plt.plot(t['strike'], t['opt_price'])
        # plt.scatter(t['strike'], t['opt_price'])
        # plt.plot(K, PRICE)
        # plt.show()

        t = day.loc[:, ['strike', 'impl_volatility']].copy().sort_values(['strike', 'impl_volatility']).reset_index(drop=True).groupby('strike').mean().reset_index()
        cb = CubicSpline(t['strike'], t['impl_volatility'])
        # X = np.arange(t['strike'].min(), t['strike'].max(), (t['strike'].max() - t['strike'].min()) / 200)
        X = s0 * np.arange(0.8, 1.3, 0.1)
        IV = cb(X)

        if self.par.data.opt_smooth == OptSmooth.EXT:
            m = np.concatenate([K[:,0], PRICE[:,0], rf, s0])
        if self.par.data.opt_smooth == OptSmooth.INT:
            m = np.concatenate([K, PRICE, rf, s0])
        if self.par.data.opt:
            p = day.loc[:, pred_col].iloc[0, :].values
        p = np.concatenate([p, IV])

        return m, p

    def clean_opt_year(self, year):
        # year = 2005
        df = pd.read_pickle(self.par.data.dir + f'raw/opt_{year}.p')
        df.columns = [x.lower() for x in df.columns]

        df = df.rename(columns={'t': 'T', 'call_put': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares',
                                's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                'expiration': 'exdate'})

        del df['index_name'], df['ticker']

        df['cp'] = df['cp'].astype('str')
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])

        df['dw'] = df['date'].dt.dayofweek
        df['dw'].hist(bins=25)
        plt.title('Day of week no clean')
        plt.savefig('res/data/day_week_split_c1.png')
        plt.close()

        df['T'] = (df['exdate'] - df['date']).dt.days

        ##################
        # strike
        ##################
        df['strike'] /= 1000

        cleaner = pd.DataFrame([self.count_sample(df, 'Raw')])
        ##################
        # Drop nan
        ##################

        df = df.dropna()

        cleaner = cleaner.append(self.count_sample(df, 'Drop na'))

        ##################
        # keep friday only
        ##################
        df['day'] = df['date'].dt.dayofweek

        # keep only friday to friday
        df = df.loc[df['day'] == 4, :]

        cleaner = cleaner.append(self.count_sample(df, 'Keep only fridays'))

        ##################
        # keep only last friday
        ##################

        # keep only last friday of month
        df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
        t = df.groupby('month')['date'].transform('max')
        ind = df['date'] == t
        ind.mean()
        df = df.loc[ind, :]
        cleaner = cleaner.append(self.count_sample(df, 'Keep last Friday'))

        ##################
        # check only one maturity per day
        ##################

        assert df.groupby(['gvkey', 'date'])['T'].std().max() == 0, 'Problem, multiple maturity on one day tic'

        ##################
        # remove wrong big and ask
        ##################
        ind = df['o_ask'] > 0
        df = df.loc[ind, :]
        ind = df['o_bid'] > 0
        df = df.loc[ind, :]

        ind = df['o_bid'] < df['o_ask']
        df = df.loc[ind, :]

        cleaner = cleaner.append(self.count_sample(df, 'Set ask>0, bid>0 and bid<ask'))

        ##################
        # remove pure duplicates
        ##################

        ind = df[['strike', 'cp', 'T', 'date', 'o_ask', 'o_bid', 'gvkey']].duplicated()
        df = df.loc[~ind, :]

        cleaner = cleaner.append(self.count_sample(df, 'Remove pure duplicated entries'))

        ##################
        # remove dupplicate in all but price
        ##################
        df['opt_price'] = (df['o_bid'] + df['o_ask']) / 2
        COL = ['cp', 'date', 'T', 'strike', 'gvkey']
        df = df.sort_values(COL + ['opt_price'])
        ind = df[COL].duplicated(keep='last')
        df = df.loc[~ind, :]

        cleaner = cleaner.append(self.count_sample(df, 'Remove pure with all but price'))

        ##################
        # drop in the money options
        ##################
        df.head()
        ind = (df['cp'] == 'C') & (df['strike'] < df['S'])
        df = df.loc[~ind, :]
        ind = (df['cp'] == 'P') & (df['strike'] > df['S'])
        df = df.loc[~ind, :]

        cleaner = cleaner.append(self.count_sample(df, 'Drop OTM'))

        ##################
        # remove strike arbitage options
        ##################
        df['s_sort'] = df['strike']
        df.loc[df['cp'] == 'P', 's_sort'] = -df.loc[df['cp'] == 'P', 's_sort']

        df = df.sort_values(['gvkey', 'date', 'cp', 's_sort']).reset_index(drop=True)
        df['p_lag'] = df.groupby(['gvkey', 'date', 'cp'])['opt_price'].shift(1)
        df['p_lag'] = df.groupby(['gvkey', 'date', 'cp'])['p_lag'].fillna(method='bfill')
        df['p_lag'] = df.groupby(['gvkey', 'date', 'cp'])['p_lag'].cummin()
        df['t'] = df['opt_price'] <= df['p_lag']
        df['t'] = df['opt_price'] <= df['p_lag']
        df.loc[pd.isna(df['p_lag']), 't'] = True
        df = df.loc[df['t'], :]
        del df['t'], df['p_lag'], df['s_sort']

        cleaner = cleaner.append(self.count_sample(df, f'Remove strike arbitrage'))

        ##################
        # remove if spread is too large
        ##################
        # max distance
        df = df.sort_values(['gvkey', 'date', 'strike']).reset_index(drop=True)
        df['t'] = df.groupby(['gvkey', 'date'])['opt_price'].diff().abs()
        df['t'] = df.groupby(['gvkey', 'date'])['t'].transform('max')
        ind = (df['t'] > df['S'] * 0.2) & (df['t'] > 10)
        df = df.loc[~ind, :]
        cleaner = cleaner.append(self.count_sample(df, f'Remove days with spread larger than max(0.2*S,10)'))
        del df['t']
        #################
        # remove  if less than x different option strike price
        ##################

        min_nb_strike = 10
        t = df.groupby(['date', 'gvkey'])['strike'].nunique()
        df = df.merge(t.reset_index().rename(columns={'strike': 'nb_distinct'}))

        ind = df['nb_distinct'] >= min_nb_strike

        df = df.loc[ind, :]
        del df['nb_distinct']

        cleaner = cleaner.append(self.count_sample(df, f'Set minimum {min_nb_strike} diff. strikes per date/tic'))

        ##################
        # remove if volume is too low
        ##################

        # days with rule --> at least 84.5 different options per maturity
        t = df.groupby(['date', 'gvkey'])['volume'].sum()
        df = df.merge(t.reset_index().rename(columns={'volume': 'vol_tot'}))
        ind = df['vol_tot'] >= 20
        df = df.loc[ind, :]
        del df['vol_tot']

        cleaner = cleaner.append(self.count_sample(df, 'Set min sum(volume)>=20'))

        t = self.load_all_price()[['gvkey', 'date', 'S0', 'S_T', 'ret']]

        # transform gvkey to correct format
        df['gvkey'] = df['gvkey'].apply(lambda x: str(x)[:6])
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
        df = df.loc[~pd.isna(df['gvkey']), :]
        df['gvkey'] = df['gvkey'].astype(int)

        df = df.merge(t, on=['gvkey', 'date'])
        cleaner = cleaner.append(self.count_sample(df, 'Computed returns'))

        print('#' * 10, 'final cleaner of year', year, '#' * 10)
        print(cleaner)
        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_{year}.p')
        return df, cleaner

    def clean_opt_all(self):
        df = []
        cleaner = []
        for y in range(2003, 2021):
            d, cl = self.clean_opt_year(y)
            df.append(d)
            cleaner.append(cl)

        df = pd.concat(df)
        c = cleaner[0]
        for i in range(1, len(cleaner)):
            c += cleaner[i]
        c.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_all.p')

        df.to_pickle(self.par.data.dir + f'{self.name}/int/opt.p')

    def load_all_price(self, reload=False):
        if reload:
            L = [x for x in os.listdir(self.par.data.dir + 'raw') if 'price_' in x]
            df = []
            for l in L:
                df.append(pd.read_pickle(self.par.data.dir + 'raw/' + l))
            df = pd.concat(df)
            df.columns = [x.lower() for x in df.columns]
            df = df.rename(columns={'gv_key': 'gvkey', 'adjustment_factor_2': 'adj', 's_close': 'S0'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
            df['S0'] = df['S0'].abs()

            T = 1
            df['S_T'] = df.groupby(['gvkey'])['S0'].shift(-T)
            df['adj_T'] = df.groupby(['gvkey'])['adj'].shift(-T)
            df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
            df['ret_1'] = np.log(df['S_T'] / df['S0'])

            T = 28
            df['S_T'] = df.groupby(['gvkey'])['S0'].shift(-T)
            df['adj_T'] = df.groupby(['gvkey'])['adj'].shift(-T)
            df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
            df['ret'] = np.log(df['S_T'] / df['S0'])
            # df=df.dropna()

            df['gvkey'] = df['gvkey'].apply(lambda x: str(x)[:6])
            df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
            df = df.loc[~pd.isna(df['gvkey']), :]
            df['gvkey'] = df['gvkey'].astype(int)

            df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)

            df.to_pickle(self.par.data.dir + f'{self.name}/int/price.p')
        else:
            df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/price.p')
        return df

    def count_sample(self, df, n=''):
        month_not_in = 'month' not in df.columns
        if month_not_in:
            try:
                df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            except:
                df['month'] = df['opt_date'].dt.year * 100 + df['opt_date'].dt.month

        r = {'Unique row': df.shape[0]}
        try:
            r['unique date'] = df['date'].unique().shape[0]
        except:
            r['unique date'] = df['opt_date'].unique().shape[0]
        r['unique month'] = df['month'].unique().shape[0]

        r['unique firm'] = df['gvkey'].unique().shape[0]

        try:
            r['unique day/firm'] = df[['date', 'gvkey']].drop_duplicates().shape[0]
        except:
            r['unique day/firm'] = df[['opt_date', 'gvkey']].drop_duplicates().shape[0]

        r['unique month/firm'] = df[['month', 'gvkey']].drop_duplicates().shape[0]

        if month_not_in:
            del df['month']

        return pd.Series(r, name=n)

    def load_compustat(self, frequency='quarterly'):
        df = pd.read_pickle(self.par.data.dir + f'raw/compustat_{frequency}.p')
        df.columns = [x.lower() for x in df.columns]
        df = df.rename(columns={'atq': 'asset', 'ibc': 'income', 'ibq': 'income', 'dlttq': 'debt', 'datadate': 'date'})
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')
        df = df.loc[~pd.isna(df['gvkey']), :]
        return df

    def load_opt(self):
        df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/opt.p')
        return df

    def load_pred_crsp_only(self, reload=False):
        if reload:
            ##################
            # recreating the appendix C variable of the Kadan Tang paper (RFS)
            ##################
            var_list = ['mkt_cap', 'mom_f']
            ################# The monthly ones

            # df = raw.crsp
            df = self.load_all_price()

            ### mkt cap
            df['mkt_cap'] = np.log(1000 * df['shares_outstanding'] * df['S0'])
            ### momentum
            df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)

            df['r'] = np.log(df['ret'] + 1)
            df.index = df['date']
            t = df.groupby('gvkey')['r'].rolling(12).sum().reset_index().rename(columns={'r': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)
            final = df[['gvkey', 'date'] + var_list].copy()

            ################ The daily ones
            #### amihud's liquidity group per month
            df = self.load_all_price().sort_values(['gvkey', 'date']).reset_index(drop=True)

            df['liq_m'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby(['gvkey', 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby('gvkey')['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final['month'] = final['date'].dt.month + final['date'].dt.year * 100
            t['month'] = t['date'].dt.month + t['date'].dt.year * 100
            del t['date']
            final = final.merge(t)
            #### amihud's liquidity group per year
            print(final.shape)
            df = self.load_all_price().sort_values(['gvkey', 'date']).reset_index(drop=True)
            df['liq_y'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby(['gvkey', 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby('gvkey')['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')
            var_list.append('liq_m')
            var_list.append('liq_y')
            ##################
            # adding the one got through regression
            ##################
            final['month'] = final['date'].dt.year * 100 + final['date'].dt.month
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            df['date'] = df['date']-pd.DateOffset(months=1)
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            df['date'] = df['date']-pd.DateOffset(months=1)
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            df['date'] = df['date']-pd.DateOffset(months=1)
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
            df['date'] = df['date']-pd.DateOffset(months=1)
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            var_list.append('beta_monthly')
            var_list.append('beta_daily')
            var_list.append('id_risk_monthly')
            var_list.append('id_risk_daily')
            ##################
            # add macro factor
            ##################
            raw = RawData(par=self.par, freq='daily')
            final = final.merge(raw.ff)
            var_list = var_list + list(raw.ff.iloc[:, 1:].columns)
            final = final[['gvkey', 'date'] + var_list].copy()
            final.to_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp.p')
        else:
            final = pd.read_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp.p')
        return final

    def load_pred_compustat_and_crsp(self, reload=False):
        if reload:
            ##################
            # recreating the appendix C variable of the Kadan Tang paper (RFS)
            ##################
            var_list = ['mkt_cap', 'btm', 'mom_f', 'inv', 'prof']
            ################# The monthly ones
            # df = raw.crsp
            df = self.load_all_price()

            ### mkt cap
            df['mkt_cap'] = np.log(1000 * df['shares_outstanding'] * df['S0'])
            ### momentum
            df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)

            df['r'] = np.log(df['ret'] + 1)
            df.index = df['date']
            t = df.groupby('gvkey')['r'].rolling(12).sum().reset_index().rename(columns={'r': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)

            ### Book to market
            df['year'] = df['date'].dt.year
            comp = self.load_compustat('quarterly').sort_values(['gvkey', 'date']).reset_index(drop=True)
            comp['year'] = comp['date'].dt.year
            comp['quarter'] = comp.index
            comp['quarter'] = comp.groupby(['gvkey', 'year'])['quarter'].transform('min')
            comp['quarter'] = comp.index - comp['quarter']
            comp['last_quarter'] = comp.groupby(['gvkey', 'year'])['quarter'].transform('max')
            comp = comp.loc[comp['quarter'] == comp['last_quarter'], ['year', 'gvkey', 'asset', 'debt']].sort_values(['gvkey', 'year']).reset_index(drop=True)
            comp['asset_l'] = comp.groupby('gvkey')['asset'].shift(1)

            # df.merge(comp,on=['gvkey','year'])

            df = df.merge(comp, on=['gvkey', 'year'])

            df['btm'] = 1000 * 1000 * (df['asset'] - df['debt']) / (df['shares_outstanding'] * df['S0'])

            ### investment
            df['inv'] = df['asset'] - df['asset_l']
            ### Profitability
            comp = self.load_compustat('quarterly').loc[:, ['date', 'gvkey', 'asset', 'debt']].sort_values(['gvkey', 'date']).reset_index(drop=True)
            comp['book_equity'] = comp['asset'] - comp['debt']
            comp['book_equity'] = comp.groupby('gvkey')['book_equity'].shift(1)

            df = df.merge(comp, how='left')
            df.index = df['date']
            t = df.groupby('gvkey').apply(lambda x: x['book_equity'].fillna(method='ffill').fillna(method='bfill')).reset_index().drop_duplicates()
            df = df.reset_index(drop=True).drop(columns='book_equity')
            df = df.merge(t, how='left')
            comp = self.load_compustat('yearly').drop_duplicates()
            comp['year'] = comp['date'].dt.year
            comp = comp.drop(columns='date')
            df = df.merge(comp)
            df['prof'] = df['income'] / df['book_equity']
            final = df[['gvkey', 'date'] + var_list].copy()
            print(pd.isna(final).mean())

            ################ The daily ones
            #### amihud's liquidity group per month
            df = self.load_all_price().sort_values(['gvkey', 'date']).reset_index(drop=True)

            df['liq_m'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby(['gvkey', 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby('gvkey')['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final['month'] = final['date'].dt.month + final['date'].dt.year * 100
            t['month'] = t['date'].dt.month + t['date'].dt.year * 100
            del t['date']
            final = final.merge(t)
            #### amihud's liquidity group per year
            print(final.shape)
            df = self.load_all_price().sort_values(['gvkey', 'date']).reset_index(drop=True)
            df['liq_y'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby(['gvkey', 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby('gvkey')['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')
            var_list.append('liq_m')
            var_list.append('liq_y')
            ##################
            # adding the one got through regression
            ##################
            final['month'] = final['date'].dt.year * 100 + final['date'].dt.month
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
            df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            del df['date']
            final = final.merge(df)

            var_list.append('beta_monthly')
            var_list.append('beta_daily')
            var_list.append('id_risk_monthly')
            var_list.append('id_risk_daily')
            ##################
            # add macro factor
            ##################
            raw = RawData(par=self.par, freq='daily')
            final = final.merge(raw.ff)
            var_list = var_list + list(raw.ff.iloc[:, 1:].columns)
            final = final[['gvkey', 'date'] + var_list].copy()
            final.to_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp_compustat.p')
        else:
            final = pd.read_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp_compustat.p')
        return final

    def get_beta(self, freq='daily'):
        print('#' * 50)
        print('Start Beta', freq)
        print('#' * 50)
        if freq == 'monthly':
            r = 12
        else:
            r = 12

        raw = RawData(self.par, freq)
        try:
            df = raw.ff.merge(self.load_all_price(False), how='inner', on='date')
        except:
            df = raw.ff.merge(self.load_all_price(True), how='inner', on='date')
        if freq == 'monthly':
            ret_v = 'ret'
        else:
            ret_v = 'ret_1'

        df['one'] = 1
        C = ['one', 'mkt-rf', 'smb', 'hml', 'mom'] + ['rf', 'gvkey', 'date'] + [ret_v]
        df = df[C].dropna()

        def get_delta(x):
            if x.shape[0] > r / 4:
                return sm.OLS(x[ret_v] - x['rf'], x[['one', 'mkt-rf']]).fit().params['mkt-rf']
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('gvkey').apply(get_delta)
            t.name = date
            res.append(t)
            date += pd.DateOffset(days=1)
            # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.day == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'gvkey', 'value': f'beta_{freq}'})
        return t

    def get_list_permno_from_crsp(self):
        df = pd.read_csv(self.par.data.dir + 'raw/crsp_monthly.csv')
        df['PERMNO'].drop_duplicates().to_csv('data/permno.txt', index=False, header=False)

    def get_list_secid_from_crsp(self):
        df = pd.read_csv(self.par.data.dir + 'raw/crsp_to_opt.csv')
        df['secid'].drop_duplicates().to_csv('data/secid.txt', index=False, header=False)

    def get_id_risk(self, freq='monthly'):
        print('#' * 50)
        print('Start id risk', freq)
        print('#' * 50)
        if freq == 'monthly':
            r = 12
        else:
            r = 12
        raw = RawData(self.par, freq)

        try:
            df = raw.ff.merge(self.load_all_price(False), how='inner', on='date')
        except:
            df = raw.ff.merge(self.load_all_price(True), how='inner', on='date')

        if freq == 'monthly':
            ret_v = 'ret'
        else:
            ret_v = 'ret_1'

        df['one'] = 1
        C = ['one', 'mkt-rf', 'smb', 'hml', 'mom'] + ['rf', 'gvkey', 'date'] + [ret_v]
        df = df[C].dropna()

        def get_std(x):
            if x.shape[0] > r / 4:
                return np.std(sm.OLS(x[ret_v] - x['rf'], x[['one', 'mkt-rf', 'smb', 'hml', 'mom']]).fit().resid)
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')

        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('gvkey').apply(get_std)
            t.name = date
            res.append(t)
            date += pd.DateOffset(days=1)
            # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.day == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'gvkey', 'value': f'id_risk_{freq}'})
        return t

    def gen_delta(self):
        df = self.get_beta('monthly')
        df.to_pickle(self.par.data.dir + f'int_gen/beta_m.p')
        df = self.get_beta('daily')
        df.to_pickle(self.par.data.dir + f'int_gen/beta_daily.p')

    def gen_id_risk(self):
        df = self.get_id_risk('monthly')
        df.to_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
        df = self.get_id_risk('daily')
        df.to_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')

    def gen_all_int(self):
        self.gen_delta()
        self.gen_id_risk()

    def pre_process_all(self):
        # pre-process option
        self.load_all_price(reload=True)
        self.clean_opt_all()

        if self.par.data.crsp:
            if self.par.data.comp:
                self.load_pred_compustat_and_crsp(reload=True)
            else:
                self.load_pred_crsp_only(reload=True)

        self.create_a_dataset()


par = Params()
# par.data.crsp=False
self = Data(par)
self.gen_all_int()
# freq='daily'
# self.pre_process_all()
# df=self.load_all_price()
#
# df['S0'].min()
# df[['stock_key']].drop_duplicates().astype(int).sort_values('stock_key').to_csv('data/secid_list.txt', index=False, header=False)
# self.pre_process_all()
# self.gen_all_int()
# self.load_final()
# self.move_shuffle()
# self.test_m_df
