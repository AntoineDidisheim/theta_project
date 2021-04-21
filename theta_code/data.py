import pandas as pd
import numpy as np
from parameters import *
import os
import statsmodels.api as sm

import socket

if socket.gethostname() !='work':
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from Econ import Econ
from scipy import optimize as opti
import multiprocessing as mp
from scipy import stats
from pandarallel import pandarallel


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
        self.id_col = 'gvkey'

        ##################
        # set name
        ##################
        self.name = f'd_{self.par.data.min_opt_per_day}'
        if self.par.data.opt:
            self.name += 'Opt'
        if self.par.data.comp:
            self.name += 'Comp'
        if self.par.data.crsp:
            self.name += 'Crsp'
        if self.par.data.mw:
            self.name += 'MW'
        if self.par.data.last_friday:
            self.name += 'LF'

        self.name += str(self.par.data.opt_smooth.name)

        ##################
        # make dir
        ##################
        self.make_dir(f"{self.par.data.dir}int_gen")
        self.make_dir(f"{self.par.data.dir}{self.name}/side")
        self.make_dir(f'{self.par.data.dir}{self.name}/int/')

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def set_year_test(self, year):

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

        print(f'Set training year {year}',flush=True)

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
        print(f'Set shuffle {self.shuffle_id}',flush=True)

    def load_final(self):
        # finally load all all
        self.m_df = pd.read_pickle(self.par.data.dir + f'{self.name}/m_df.p').reset_index(drop=True)
        self.p_df = pd.read_pickle(self.par.data.dir + f'{self.name}/pred_df.p').reset_index(drop=True)
        self.label_df = pd.read_pickle(self.par.data.dir + f'{self.name}/label_df.p').reset_index(drop=True)

        # add the transofrmed return
        self.label_df['log_ret'] = self.label_df['ret']
        self.label_df['normal_ret'] = np.exp(self.label_df['log_ret']) - 1

        if self.par.data.ret == ReturnType.RET:
            self.label_df['ret'] = self.label_df['normal_ret']
        if self.par.data.ret == ReturnType.LOG:
            self.label_df['ret'] = self.label_df['log_ret']



        if self.par.data.max_ret>-1:
            ind = (self.label_df['ret']>=self.par.data.min_ret) & (self.label_df['ret']<=self.par.data.max_ret)
            self.m_df = self.m_df.loc[ind,:].reset_index(drop=True)
            self.p_df = self.p_df.loc[ind,:].reset_index(drop=True)
            self.label_df = self.label_df.loc[ind,:].reset_index(drop=True)

        if self.par.data.hist_theta:
            hist = self.historical_theta()
            t=self.label_df.merge(hist,how='left')['theta']
            self.p_df['theta_hist']=t






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

        df = pred.merge(opt, on=[self.id_col, 'date'])
        cleaner = cleaner.append(self.count_sample(df, 'Merge with predictors'))

        df = df.dropna()
        cleaner = cleaner.append(self.count_sample(df, 'Drop missing predictors'))
        print(cleaner,flush=True)

        ind = df['cp'] == 'C'
        mc = df.loc[ind, :].groupby(['date', self.id_col])['ret'].count().max()
        ind = df['cp'] == 'P'
        mp_ = df.loc[ind, :].groupby(['date', self.id_col])['ret'].count().max()
        print('Max nb call/put in sample:', mc, mp_,flush=True)
        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_after_merge.csv')
        return df

    def create_a_dataset(self):
        print('In create dataset', flush=True)

        if self.par.data.crsp:
            if self.par.data.comp:
                pred = self.load_pred_compustat_and_crsp()
            else:
                pred = self.load_pred_crsp_only()

        # specify here the list of DataType which do not need to be merge with any dataset
        if self.par.data.opt and (not self.par.data.comp) and (not self.par.data.crsp):
            df = self.load_opt()
            raw = RawData(self.par, 'daily')
            df = df.merge(raw.ff[['date', 'rf']])
            pred_col = []
        else:
            df = self.load_and_merge_pred_opt()
            pred_col = list(pred.drop(columns=[self.id_col, 'date']).columns)

        if self.par.data.mw:
            mw = self.marting_wagner_return()

            ind = mw[['date', self.id_col]].duplicated(keep='first')
            mw = mw.loc[~ind, :]
            df = df.merge(mw, how='left')
            pred_col.append('MW')

        ##################
        # process a day
        ##################
        # select a day
        t = df[['date', self.id_col, 'ret']].drop_duplicates().reset_index(drop=True)
        M = []
        P = []
        i = 25000
        i = 2

        if 'rf_x' in df.columns:
            df['rf'] = df['rf_x'].values
            del df['rf_x']


        RF = self.load_rf()
        for i in range(1500,t.shape[0]):
        # for i in range(t.shape[0]):
            id = t[['date', self.id_col]].iloc[i, :]
            ind = (df['date'] == id['date']) & (df[self.id_col] == id[self.id_col])
            day = df.loc[ind, :]
            day['delta']
            day=day.loc[day['delta'].abs()<=0.5,:]

            if day.loc[:, ['strike', 'opt_price', 'impl_volatility']].drop_duplicates().shape[0]>1:
                m, p = self.pre_process_day(day, pred_col, RF)

                M.append(m)
                P.append(p)
            else:
                print(f'### Skip {i}, not enough points')

            if i % 100 == 0:
                print(i, '/', t.shape[0],flush=True)


        # ### try apply
        # def create_function(id):
        #     # id = t[['date', self.id_col]].iloc[i, :]
        #     ind = (df['date'] == id['date']) & (df[self.id_col] == id[self.id_col])
        #     day = df.loc[ind, :]
        #     day = day.loc[day['delta'].abs() <= 0.5, :]
        #     m, p = self.pre_process_day(day, pred_col, RF)
        #     return m, p
        #
        #
        # pandarallel.initialize(progress_bar=True,nb_workers=20)
        # r=t.a(lambda x: create_function(x),axis=1)
        # M=r.apply(lambda x: x[0]).values.tolist()
        # P=r.apply(lambda x: x[1]).values.tolist()

        ### end apply

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

    def pre_process_day(self, day, pred_col, RF):


        s0 = np.array(day.loc[:, 'S0'].iloc[0]).reshape(1)
        try:
            rf = RF.loc[RF.iloc[:, 0] == day['date'].iloc[0], 1].iloc[0]
        except:
            rf = np.array(day.loc[:, 'rf'].iloc[0]).reshape(1)
            print('missing rf',flush=True)

        t = day.loc[:, ['strike', 'opt_price','impl_volatility']].copy().sort_values(['strike', 'opt_price','impl_volatility']).reset_index(drop=True).groupby('strike').mean().reset_index()



        # cb = CubicSpline(t['strike'], t['opt_price'])
        bound = (t['impl_volatility'].iloc[0], t['impl_volatility'].iloc[-1])
        cb = interp1d(t['strike'], t['impl_volatility'], bounds_error=False, fill_value=bound)
        if self.par.data.opt_smooth == OptSmooth.EXT:
            K = np.linspace(s0 * 1/3, s0 * 3, Constant.GRID_SIZE)
        if self.par.data.opt_smooth == OptSmooth.INT:
            K = np.linspace(t['strike'].min(), t['strike'].max(), Constant.GRID_SIZE)
        assert len(K) == Constant.GRID_SIZE, 'Problem with the linespace'


        IV = cb(K)

        def BlackScholes_price(S, r, sigma, K):
            dt = 28/365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S>K] = pr_put[S>K]
            return pr

        S = day['S'].iloc[0]


        PRICE=BlackScholes_price(S,rf,IV,K)
        #
        #
        # plt.plot(t['strike'], t['impl_volatility'])
        # plt.scatter(t['strike'], t['impl_volatility'])
        # plt.plot(K, IV)
        # plt.show()
        #
        # #
        # plt.plot(t['strike'], t['opt_price'])
        # plt.scatter(t['strike'], t['opt_price'])
        # plt.plot(K, PRICE)
        # plt.show()


        t = day.loc[:, ['strike', 'impl_volatility']].copy().sort_values(['strike', 'impl_volatility']).reset_index(drop=True).groupby('strike').mean().reset_index()
        cb = CubicSpline(t['strike'], t['impl_volatility'])
        # X = np.arange(t['strike'].min(), t['strike'].max(), (t['strike'].max() - t['strike'].min()) / Constant.GRID_SIZE)
        X = s0 * np.arange(0.8, 1.3, 0.1)
        IV = cb(X)

        if self.par.data.opt_smooth == OptSmooth.EXT:
            m = np.concatenate([K[:, 0], PRICE[:, 0], [rf], s0])
        if self.par.data.opt_smooth == OptSmooth.INT:
            m = np.concatenate([K, PRICE, [rf], s0])
        p = day.loc[:, pred_col].iloc[0, :].values

        if self.par.data.opt:
            p = np.concatenate([p, IV])

        return m, p

    def clean_opt_year(self, year, input_df=None):
        # year = 2005

        df = pd.read_pickle(self.par.data.dir + f'raw/opt_{year}.p')

        df.columns = [x.lower() for x in df.columns]
        del df['gv_key']

        df = df.rename(columns={'t': 'T', 'call_put': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares',
                                's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                'expiration': 'exdate'})

        try:
            del df['index_name'], df['ticker']
        except:
            print('index name and ticker already del',flush=True)

        df['cp'] = df['cp'].astype('str')
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])

        df['dw'] = df['date'].dt.dayofweek
        df['dw'].hist(bins=25)
        # plt.title('Day of week no clean')
        # plt.savefig('res/data/day_week_split_c1.png')
        # plt.close()

        df['T'] = (df['exdate'] - df['date']).dt.days

        ##################
        # strike
        ##################
        df['strike'] /= 1000
        df.loc[pd.isna(df['gvkey']),:]

        cleaner = pd.DataFrame([self.count_sample(df, 'Raw')])
        ##################
        # Drop nan
        ##################

        df = df.dropna()

        cleaner = cleaner.append(self.count_sample(df, 'Drop na'))

        ##################.
        # keep friday only
        ##################
        df['day'] = df['date'].dt.dayofweek

        # keep only friday to friday
        df = df.loc[df['day'] == 4, :]

        cleaner = cleaner.append(self.count_sample(df, 'Keep only fridays'))

        ##################
        # keep only last friday
        ##################
        if self.par.data.last_friday:
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
        df['t'] = df.groupby(['date'])['T'].transform('std')
        t = df.loc[df['t'] > 0, :]
        t.groupby('date').max()

        if df.groupby([self.id_col, 'date'])['T'].std().max() == 0:
            print('All good, only one maturity per day per gvkey')
        else:
            print('Problem, multiple maturity on one day tic, fixing now')

            t = df.groupby([self.id_col, 'date', 'T'])['strike'].count().reset_index().rename(columns={'strike': 'nb'})
            t = t.sort_values([self.id_col, 'date', 'nb']).reset_index(drop=True)
            t[[self.id_col, 'date', 'T']].duplicated(keep='first').mean()
            df

            t['nb_max'] = t.groupby([self.id_col, 'date'])['nb'].transform('max')
            t['keep'] = t['nb'] == t['nb_max']
            del t['nb'], t['nb_max']
            df = df.merge(t)
            df = df.loc[df['keep'], :]
            df = df.reset_index(drop=True)
            del df['keep']

        # assert  '

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

        ind = df[['strike', 'cp', 'T', 'date', 'o_ask', 'o_bid', self.id_col]].duplicated()
        df = df.loc[~ind, :]

        cleaner = cleaner.append(self.count_sample(df, 'Remove pure duplicated entries'))

        ##################
        # remove dupplicate in all but price
        ##################
        df['opt_price'] = (df['o_bid'] + df['o_ask']) / 2
        COL = ['cp', 'date', 'T', 'strike', self.id_col]
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

        df = df.sort_values([self.id_col, 'date', 'cp', 's_sort']).reset_index(drop=True)
        df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['opt_price'].shift(1)
        df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['p_lag'].fillna(method='bfill')
        df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['p_lag'].cummin()
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
        df = df.sort_values([self.id_col, 'date', 'strike']).reset_index(drop=True)
        df['t'] = df.groupby([self.id_col, 'date'])['opt_price'].diff().abs()
        df['t'] = df.groupby([self.id_col, 'date'])['t'].transform('max')
        ind = (df['t'] > df['S'] * 0.2) & (df['t'] > 10)
        df = df.loc[~ind, :]
        cleaner = cleaner.append(self.count_sample(df, f'Remove days with spread larger than max(0.2*S,10)'))
        del df['t']
        #################
        # remove  if less than x different option strike price
        ##################

        min_nb_strike = self.par.data.min_opt_per_day
        t = df.groupby(['date', self.id_col])['strike'].nunique()
        df = df.merge(t.reset_index().rename(columns={'strike': 'nb_distinct'}))

        ind = df['nb_distinct'] >= min_nb_strike

        df = df.loc[ind, :]
        del df['nb_distinct']

        cleaner = cleaner.append(self.count_sample(df, f'Set minimum {min_nb_strike} diff. strikes per date/tic'))

        ##################
        # remove if volume is too low
        ##################

        # days with rule --> at least 84.5 different options per maturity
        t = df.groupby(['date', self.id_col])['volume'].sum()
        df = df.merge(t.reset_index().rename(columns={'volume': 'vol_tot'}))
        ind = df['vol_tot'] >= 20
        df = df.loc[ind, :]
        del df['vol_tot']

        cleaner = cleaner.append(self.count_sample(df, 'Set min sum(volume)>=20'))

        t = self.load_all_price()[[self.id_col, 'date', 'S0', 'S_T', 'ret']]

        # transform gvkey to correct format
        df[self.id_col] = df[self.id_col].apply(lambda x: str(x)[:6])
        df[self.id_col] = pd.to_numeric(df[self.id_col], errors='coerce')
        df = df.loc[~pd.isna(df[self.id_col]), :]
        df[self.id_col] = df[self.id_col].astype(int)

        df = df.merge(t, on=[self.id_col, 'date'])
        cleaner = cleaner.append(self.count_sample(df, 'Computed returns'))

        print('#' * 10, 'final cleaner of year', year, '#' * 10)
        print(cleaner,flush=True)
        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_{year}.p')
        return df, cleaner

    def clean_opt_all(self):
        df = []
        cleaner = []
        for y in range(1996, 2021):
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
            del df['gv_key']
            df = df.rename(columns={'gv_key': 'gvkey', 'adjustment_factor_2': 'adj', 's_close': 'S0'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)
            df['S0'] = df['S0'].abs()

            T = 1
            df['S_T'] = df.groupby([self.id_col])['S0'].shift(-T)
            df['adj_T'] = df.groupby([self.id_col])['adj'].shift(-T)
            df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
            df['ret_1'] = np.log(df['S_T'] / df['S0'])
            T = 28
            df['S_T'] = df.groupby([self.id_col])['S0'].shift(-T)
            df['adj_T'] = df.groupby([self.id_col])['adj'].shift(-T)
            df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
            df['ret'] = np.log(df['S_T'] / df['S0'])
            # df=df.dropna()

            df[self.id_col] = df[self.id_col].apply(lambda x: str(x)[:6])
            df[self.id_col] = pd.to_numeric(df[self.id_col], errors='coerce')
            df = df.loc[~pd.isna(df[self.id_col]), :]
            df[self.id_col] = df[self.id_col].astype(int)

            df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)

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

        r['unique firm'] = df[self.id_col].unique().shape[0]

        r['unique day/firm'] = df[['date', self.id_col]].drop_duplicates().shape[0]

        r['unique month/firm'] = df[['month', self.id_col]].drop_duplicates().shape[0]

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
            df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)

            df.index = df['date']
            t = df.groupby(self.id_col)['ret_1'].rolling(12).sum().reset_index().rename(columns={'ret_1': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)

            final = df[[self.id_col, 'date'] + var_list].copy()

            ################ The daily ones
            #### amihud's liquidity group per month
            df = self.load_all_price().sort_values([self.id_col, 'date']).reset_index(drop=True)

            df['liq_m'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby([self.id_col, 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby(self.id_col)['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final['month'] = final['date'].dt.month + final['date'].dt.year * 100
            t['month'] = t['date'].dt.month + t['date'].dt.year * 100
            del t['date']
            final = final.merge(t)
            #### amihud's liquidity group per year
            print(final.shape,flush=True)
            df = self.load_all_price().sort_values([self.id_col, 'date']).reset_index(drop=True)
            df['liq_y'] = df['ret'].abs() / df['total_volume']
            df = df.loc[df['total_volume'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby([self.id_col, 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby(self.id_col)['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')
            var_list.append('liq_m')
            var_list.append('liq_y')

            ### shifting all
            del final['month']
            for c in ['mom_f']:
                final[c] = final.groupby(self.id_col)[c].shift(1)

            ##################
            # adding the one got through regression
            ##################

            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            # df['date'] = df['date']-pd.DateOffset(months=1)
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            # df['date'] = df['date']-pd.DateOffset(months=1)
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            # df['date'] = df['date']-pd.DateOffset(months=1)
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
            # df['date'] = df['date']-pd.DateOffset(months=1)
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
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
            final = final[[self.id_col, 'date'] + var_list].copy()
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

            df.index = df['date']
            t = df.groupby('gvkey')['ret_1'].rolling(12).sum().reset_index().rename(columns={'ret_1': 'mom_f'})
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

            ### shifting all
            del final['month']
            for c in ['mom_f']:
                final[c] = final.groupby('gvkey')[c].shift(1)

            ##################
            # adding the one got through regression
            ##################
            final['month'] = final['date'].dt.year * 100 + final['date'].dt.month
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
            final = final.merge(df)

            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
            # df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
            # del df['date']
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
        print('#' * 50,flush=True)
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
        C = ['one', 'mkt-rf', 'smb', 'hml', 'mom'] + ['rf', self.id_col, 'date'] + [ret_v]
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
            t = df.loc[ind, :].groupby(self.id_col).apply(get_delta)
            t.name = date
            res.append(t)
            date += pd.DateOffset(days=1)
            # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.day == 1:
                print(date, flush=True)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'gvkey', 'value': f'beta_{freq}'})
        return t

    def load_rf(self, reload=False):
        if reload:
            df = pd.read_csv(self.par.data.dir + 'raw/rf.csv')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['one'] = 1
            final = []
            for d in df['date'].unique():
                temp = df.loc[df['date'] == d, :]

                m = KNeighborsRegressor(n_neighbors=2, weights='distance').fit(temp[['one', 'days']], temp['rate'])

                rf = m.predict(pd.DataFrame(data={'one': [1], 'days': [28]}))[0] / 100
                final.append([d, rf])

            df = pd.DataFrame(final)
            df.to_pickle(self.par.data.dir + 'raw_merge/rf.p')

        df = pd.read_pickle(self.par.data.dir + 'raw_merge/rf.p')
        return df

    def martin_wagner_var_mkt(self, reload=False):
        if reload:
            df = pd.read_csv(self.par.data.dir + 'raw/spx_opt.csv')
            price = pd.read_csv(self.par.data.dir + 'raw/spx_price.csv').rename(columns={'close': 'S'})[['date', 'S']]
            price['date'] = pd.to_datetime(price['date'], format='%Y%m%d')

            df = df.rename(columns={'t': 'T', 'cp_flag': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares', 'strike_price': 'strike',
                                    's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                    'expiration': 'exdate'})
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
            df['T'] = (df['exdate'] - df['date']).dt.days
            df[self.id_col] = df['ticker']
            df = df.merge(price)

            ##################
            # clean option
            ##################

            df = df.rename(columns={'t': 'T', 'call_put': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares',
                                    's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                    'expiration': 'exdate'})

            df['cp'] = df['cp'].astype('str')
            df['date'] = pd.to_datetime(df['date'])
            df['exdate'] = pd.to_datetime(df['exdate'])

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
            # keep only one maturity per day
            ##################
            t = df.groupby([self.id_col, 'date', 'T'])['optionid'].count().reset_index().rename(columns={'optionid': 'nb'})
            t = t.sort_values([self.id_col, 'date', 'nb']).reset_index(drop=True)
            t['keep'] = ~t[['date']].duplicated(keep='first')
            del t['nb']
            df = df.merge(t)
            df = df.loc[df['keep'], :]
            df = df.reset_index(drop=True)

            assert df.groupby([self.id_col, 'date'])['T'].std().max() == 0, 'we took care of multiple maturities'

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

            ind = df[['strike', 'cp', 'T', 'date', 'o_ask', 'o_bid', self.id_col]].duplicated()
            df = df.loc[~ind, :]

            cleaner = cleaner.append(self.count_sample(df, 'Remove pure duplicated entries'))

            ##################
            # remove dupplicate in all but price
            ##################
            df['opt_price'] = (df['o_bid'] + df['o_ask']) / 2
            COL = ['cp', 'date', 'T', 'strike', self.id_col]
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

            df = df.sort_values([self.id_col, 'date', 'cp', 's_sort']).reset_index(drop=True)
            df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['opt_price'].shift(1)
            df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['p_lag'].fillna(method='bfill')
            df['p_lag'] = df.groupby([self.id_col, 'date', 'cp'])['p_lag'].cummin()
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
            df = df.sort_values([self.id_col, 'date', 'strike']).reset_index(drop=True)
            df['t'] = df.groupby([self.id_col, 'date'])['opt_price'].diff().abs()
            df['t'] = df.groupby([self.id_col, 'date'])['t'].transform('max')
            ind = (df['t'] > df['S'] * 0.2) & (df['t'] > 10)
            df = df.loc[~ind, :]
            cleaner = cleaner.append(self.count_sample(df, f'Remove days with spread larger than max(0.2*S,10)'))
            del df['t']
            #################
            # remove  if less than x different option strike price
            ##################

            min_nb_strike = 10
            t = df.groupby(['date', self.id_col])['strike'].nunique()
            df = df.merge(t.reset_index().rename(columns={'strike': 'nb_distinct'}))

            ind = df['nb_distinct'] >= min_nb_strike

            df = df.loc[ind, :]
            del df['nb_distinct']

            cleaner = cleaner.append(self.count_sample(df, f'Set minimum {min_nb_strike} diff. strikes per date/tic'))

            ##################
            # computing var q
            ##################
            df['strike_sort'] = df['strike']
            df.loc[df['cp'] == 'P', 'strike_sort'] = -df.loc[df['cp'] == 'C', 'strike_sort']
            df = df.sort_values(['date', 'strike']).reset_index(drop=True)
            res = []
            for d in df['date'].unique():
                temp = df.loc[df['date'] == d, :].copy().reset_index(drop=True)

                def trapezoidal_integral_approx(t, y):
                    return np.sum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2.)

                S = temp['S'].iloc[0]
                v = 2 * trapezoidal_integral_approx(temp['strike'].values, temp['opt_price'].values) / S ** 2
                res.append([d, v])
            res = pd.DataFrame(res, columns=['date', 'var_q'])
            res.to_pickle(self.par.data.dir + 'raw_merge/var_q.p')
        res = pd.read_pickle(self.par.data.dir + 'raw_merge/var_q.p')

        # res.rolling()
        # res.index = res['date']
        # res.pop('date')
        # res = np.sqrt(res*np.sqrt(12))
        # res.plot()
        # plt.show()

        return res

    def marting_wagner_return(self, reload=False):
        if reload:
            try:
                sp_var = self.martin_wagner_var_mkt()
            except:
                sp_var = self.martin_wagner_var_mkt(reload=True)
            df = self.load_opt()
            pr = self.load_all_price()
            pr['mkt'] = pr['shares_outstanding'] * pr['S0']
            pr['y'] = np.exp(pr['ret']) - 1

            df = df.sort_values(['date', 'strike']).reset_index(drop=True)
            t = df[['date', self.id_col]].drop_duplicates().reset_index(drop=True)

            res = []
            for i in range(t.shape[0]):
                d = t.iloc[i, 0]
                g = t.iloc[i, 1]

                temp = df.loc[(df['date'] == d) & (df[self.id_col] == g), :].copy().reset_index(drop=True)

                def trapezoidal_integral_approx(t, y):
                    return np.sum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2.)

                S = temp['S'].iloc[0]
                v = 2 * trapezoidal_integral_approx(temp['strike'].values, temp['opt_price'].values) / S ** 2
                res.append([d, g, v])
            res = pd.DataFrame(res, columns=['date', self.id_col, 'vqs'])
            df = res.merge(sp_var)
            df = df.merge(pr[[self.id_col, 'date', 'mkt']])

            df['mkt_tot'] = df.groupby('date')['mkt'].transform('sum')
            df['w'] = df['mkt'] / df['mkt_tot']
            df['w_vqs'] = df['w'] * df['vqs']
            df['sum_w_vqs'] = df.groupby('date')['w_vqs'].transform('sum')
            df['MW'] = df['var_q'] + 0.5 * (df['vqs'] - df['sum_w_vqs'])

            df = df.merge(pr[['y', 'date', self.id_col]])
            df.dropna()

            def r2(df_):
                return 1 - ((df_['y'] - df_['MW']) ** 2).sum() / (df_['y'] ** 2).sum()

            df['year'] = df['date'].dt.year

            print(r2(df))
            print(df.groupby('year').apply(r2),flush=True)

            df[['date', self.id_col, 'MW']].to_pickle(self.par.data.dir + f'{self.name}/int/MW.p')

        df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/MW.p')
        return df

    def historical_theta(self, reload=False):
        if reload:
            temp_dir = self.par.data.dir+'temp_hist_theta/'
            os.makedirs(temp_dir,exist_ok=True)
            self.load_final()



            label = self.label_df.copy()
            ind = label.sort_values([self.id_col, 'date']).index
            m = self.m_df.copy()
            m = m.loc[ind, :].reset_index(drop=True)
            label = label.loc[ind, :].reset_index(drop=True)
            res = []
            prb_i = []
            i = 1000
            def compute_theta_for_index_i(i):
                d = label.iloc[i, 0]
                g = label.iloc[i, 1]
                save_id = str(g)+'_'+str(d).split(' ')[0].replace('-','_')
                if (save_id not in os.listdir(temp_dir)):
                    ind = (label['date'] < d) & (label['date'] <= d - pd.DateOffset(years=1)) & (label[self.id_col] == g)
                    if (ind.sum() > 26):  # ask for a minimum of 6 month data
                        temp = m.loc[ind, :].copy()
                        temp['theta'] = 0.0
                        temp = temp.values
                        y = label.loc[ind, 'log_ret'].copy()
                        y = np.mean(y)
                        def func(theta):
                            temp[:, -1] = theta
                            t = np.apply_along_axis(Econ.up_down_apply_log, axis=1, arr=temp)
                            S=temp[:,-2]
                            t[:,0] = t[:,0] / S**theta
                            t[:,1] = t[:,1] / S**theta
                            t=np.mean(t,0)
                            r=t[0]/t[1]
                            r=r - np.mean(np.log(S))

                            return np.mean(np.square(r - y))

                        b = opti.Bounds([0], [5.0])
                        try:
                            r = opti.minimize(fun=func, x0=0.5, method='trust-constr', bounds=b)

                            theta = r['x'][0]
                            func(theta)
                            m_pred = m.iloc[i, :]
                            m_pred['theta'] = theta
                            pred = Econ.g_apply_log(m_pred.values)
                            m_pred['theta'] = 1.0
                            pred_0 = Econ.g_apply_log(m_pred.values)
                        except:
                            pred = np.nan
                            theta = np.nan
                            pred_0 = np.nan
                            print(f'Opti problem with {i,d,g}',flush=True)
                            prb_i.append(i)
                    else:
                        pred = np.nan
                        theta = np.nan
                        pred_0 = np.nan


                    pd.Series({'date':d,'gvkey':g,'theta':theta,'pred':pred,'pred_0':pred_0}).to_pickle(temp_dir+save_id)
                else:
                    print(f'already done ==={i}')



                if i % 100 == 0:
                    print(i, '/', label.shape[0],flush=True)
                return i

            # p = Pool(80)





            todo=np.arange(label.shape[0]).tolist()

            for i in todo:
                compute_theta_for_index_i(i)

            # p.map(compute_theta_for_index_i,todo)


            # merge and save
            t = pd.DataFrame()
            for f in os.listdir(temp_dir):
                try:
                    t=t.append(pd.DataFrame(pd.read_pickle(temp_dir+f)).T)
                except:
                    print('load prb with', f)

            res=label.merge(t,how='left')

            def fun(x):
                try:
                    x = x.numpy()
                except:
                    pass
                return x
            res['pred']=res['pred'].apply(lambda x: fun(x))
            res['pred_0']=res['pred_0'].apply(lambda x: fun(x))
            res = res.sort_values(['gvkey','date']).reset_index(drop=True)
            res['na'] = pd.isna(res['theta'])

            gv = res['gvkey'].copy()
            res=res.groupby('gvkey').transform('ffill')
            res['gvkey']=gv

            res.to_pickle(self.par.data.dir+'theta_hist.p')

        df = pd.read_pickle(self.par.data.dir+'theta_hist.p')


        return df

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
        C = ['one', 'mkt-rf', 'smb', 'hml', 'mom'] + ['rf', self.id_col, 'date'] + [ret_v]
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
            t = df.loc[ind, :].groupby(self.id_col).apply(get_std)
            t.name = date
            res.append(t)
            date += pd.DateOffset(days=1)
            # date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.day == 1:
                print(date, flush=True)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': self.id_col, 'value': f'id_risk_{freq}'})
        return t

    # def load_all_bench_r2(self):
    #     df=self.load_all_price()
    #     df['year'] = df['date'].dt.year
    #     df.groupby('year').mean()
    #


    def create_good_iv(self):
        df = self.load_opt()
        ind = (df['impl_volatility']>0) & (df['delta'].abs()<0.5)
        df = df.loc[ind,:]

        rf = self.load_rf()
        rf.columns = ['date','rf']
        df=df.merge(rf)

        def BlackScholes_price(sigma, S, r, K):
            dt = 28/365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S>K] = pr_put[S>K]
            return pr



        def BlackScholes_error(sigma, S, r, K,mid):
            dt = 28/365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S>K] = pr_put[S>K]
            return np.square(pr-mid)
        def find(d):
            x_init =d['impl_volatility']
            S = d['S0']
            r = d['rf']
            K = d['strike']
            pr = d['opt_price']
            try:
                res=opti.minimize(BlackScholes_error,x_init,args=(S,r,K,pr))
                iv=res.x
            except:
                iv=np.nan
            return iv[0]


        pandarallel.initialize(progress_bar=True)
        r= df.parallel_apply(find,axis=1)
        # r = pd.read_pickle('temp.p')
        # r = r.values
        # r.shape
        df['impl_volatility'] =r
        df.to_pickle(self.par.data.dir + f'{self.name}/int/opt.p')

        p=BlackScholes_price(df['impl_volatility'],df['S0'],df['rf'],df['strike'])
        print('Error', (p-df['opt_price']).abs().mean())







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
        self.load_all_price(reload=True)
        self.gen_delta()
        self.gen_id_risk()

    def pre_process_all(self):
        # # pre-process option
        self.load_all_price(reload=True)
        self.clean_opt_all()
        self.martin_wagner_var_mkt(reload=True)
        self.marting_wagner_return(reload=True)
        print('start',flush=True)

        if self.par.data.crsp:
            if self.par.data.comp:
                self.load_pred_compustat_and_crsp(reload=True)
            else:
                self.load_pred_crsp_only(reload=True)

        # self.historical_theta(reload=True)
        self.create_a_dataset()

#

# # self.create_a_dataset()
# self.pre_process_all()
# # self.gen_all_int()

# self.historical_theta(reload=True)

# #
# if self.par.data.crsp:
#     if self.par.data.comp:
#         self.load_pred_compustat_and_crsp(reload=True)
#     else:
#         self.load_pred_crsp_only(reload=True)
#
# self.create_a_dataset()


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
