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
        crsp = pd.read_csv(par.data.dir + f'raw/crsp_{freq}.csv')
        # crsp = pd.read_csv(par.data.dir+'crsp_monthly.csv')
        crsp['date'] = pd.to_datetime(crsp['date'], format='%Y%m%d')
        crsp = crsp.rename(columns={'PRC': 'price', 'RET': 'ret', 'VOL': 'vol', 'SHROUT': 'share', 'TICKER': 'tic', 'CFACPR': 'adj'})
        del crsp['ACPERM'], crsp['ACCOMP'], crsp['NWPERM']
        crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
        crsp['tic'] = crsp['tic'].astype('str')
        crsp = crsp.dropna()
        if freq == 'monthly':
            crsp['crsp_date'] = crsp['date'].values
            crsp['date'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
            crsp['date'] = pd.to_datetime(crsp['date'], format='%Y%m')
        self.crsp = crsp

        compustat = pd.read_csv(par.data.dir + 'raw/compustat_quarterly.csv')
        compustat = compustat.rename(columns={'datadate': 'date', 'fyearq': 'year', 'fqtr': 'quarter', 'atq': 'asset',
                                              'dlttq': 'debt', 'ibq': 'income', 'mkvaltk': 'mkt_cap',
                                              'LINKENDDT': 'd_end', 'LINKDT': 'd_start'}) \
            .drop(columns=['GVKEY', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat'])
        compustat['date'] = pd.to_datetime(compustat['date'], format='%Y%m%d')

        compustat.loc[compustat['d_end'].max() == compustat['d_end'], 'd_end'] = compustat.loc[compustat['d_end'] != 'E', 'd_end'].max()
        compustat['d_start'] = pd.to_datetime(compustat['d_end'], format='%Y%m%d')
        compustat['d_end'] = pd.to_datetime(compustat['d_end'], format='%Y%m%d')

        if freq == 'monthly':
            compustat['date'] = compustat['date'].dt.year * 100 + compustat['date'].dt.month
            compustat['date'] = pd.to_datetime(compustat['date'], format='%Y%m')
        # del compustat['tic']
        compustat = compustat.rename(columns={'LPERMNO': 'PERMNO'})

        self.compustat = compustat

        compustat_yearly = pd.read_csv(par.data.dir + 'raw/compustat_yearly.csv')
        compustat_yearly = compustat_yearly.rename(columns={'datadate': 'date', 'fyearq': 'year', 'fqtr': 'quarter', 'ib': 'income', 'ticker': 'tic'
            , 'fyear': 'year', 'LINKENDDT': 'd_end', 'LINKDT': 'd_start'}) \
            .drop(columns=['GVKEY', 'indfmt', 'consol', 'date', 'popsrc', 'curcd', 'datafmt', 'costat'])

        compustat_yearly = compustat_yearly.dropna()
        compustat_yearly['year'] = compustat_yearly['year'].astype(int)

        compustat_yearly = compustat_yearly.rename(columns={'LPERMNO': 'PERMNO'})
        compustat_yearly.loc[compustat_yearly['d_end'].max() == compustat_yearly['d_end'], 'd_end'] = compustat_yearly.loc[compustat_yearly['d_end'] != 'E', 'd_end'].max()
        compustat_yearly['d_start'] = pd.to_datetime(compustat_yearly['d_end'], format='%Y%m%d')
        compustat_yearly['d_end'] = pd.to_datetime(compustat_yearly['d_end'], format='%Y%m%d')

        self.compustat_yearly = compustat_yearly

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
        # make dir
        ##################
        self.make_dir(f"{self.par.data.dir}int_gen")
        self.make_dir(f"{self.par.data.dir}side")
        self.make_dir(f'{self.par.data.dir}{self.par.data.dtype.name}/int/')

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

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
        self.m_df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/m_df.p').reset_index(drop=True)
        self.p_df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/pred_df.p').reset_index(drop=True)
        self.label_df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/label_df.p').reset_index(drop=True)
        # convert ret to log ret
        self.label_df['ret'] = np.log(self.label_df['ret'] + 1)

        # ind = (self.label_df['ret'] <= 1.0) & (self.label_df['ret'] >= -1.0)
        # self.label_df = self.label_df.loc[ind, :].reset_index(drop=True)
        # self.m_df = self.m_df.loc[ind, :].reset_index(drop=True)
        # self.p_df = self.p_df.loc[ind, :].reset_index(drop=True)

        for c in self.p_df.columns:
            self.p_df.loc[:, c] = (self.p_df[c] - self.p_df[c].mean()) / self.p_df[c].std()

        # self.p_df = self.p_df.clip(-3.0, 3.0)

    def load_and_merge_pred_opt(self):
        cleaner = pd.read_pickle(self.par.data.dir + 'side/cleaner_final.p')
        opt = self.load_opt()

        if self.par.data.dtype in [DataType.COMP_CRSP_OPTION_1,DataType.COMP_CRSP_1]:
            pred = self.load_pred_compustat_and_crsp()
        if self.par.data.dtype in [DataType.CRSP_1,DataType.CRSP_OPTION_1]:
            pred = self.load_pred_crsp_only()

        pred['month'] = pred['crsp_date'].dt.year * 100 + pred['crsp_date'].dt.month


        df = pred.merge(opt, on=['PERMNO', 'month'])
        cleaner = cleaner.append(self.count_sample(df, 'Merge with predictors'))

        df = df.dropna()
        cleaner = cleaner.append(self.count_sample(df, 'Drop missing predictors'))
        print(cleaner)

        ind = df['cp'] == 'C'
        mc = df.loc[ind, :].groupby(['opt_date', 'PERMNO'])['ret'].count().max()
        ind = df['cp'] == 'P'
        mp = df.loc[ind, :].groupby(['opt_date', 'PERMNO'])['ret'].count().max()
        print('Max nb call/put in sample:', mc, mp)
        return df

    def create_a_dataset(self):

        if self.par.data.dtype in [DataType.COMP_CRSP_1,DataType.COMP_CRSP_OPTION_1]:
            pred = self.load_pred_compustat_and_crsp()
        if self.par.data.dtype in [DataType.CRSP_OPTION_1,DataType.CRSP_1]:
            pred = self.load_pred_crsp_only()

        # specify here the list of DataType which do not need to be merge with any dataset
        if self.par.data.dtype in [DataType.OPTION_1]:
            df = self.load_opt()
            raw = RawData(self.par,'daily')
            df = df.merge(raw.ff[['date','rf']].rename(columns={'date':'opt_date'}))
            pred_col = []
        else:
            df = self.load_and_merge_pred_opt()
            pred_col = list(pred.drop(columns=['PERMNO', 'date', 'crsp_date']).columns)
        ##################
        # process a day
        ##################
        # select a day
        t = df[['opt_date', 'PERMNO', 'ret']].drop_duplicates().reset_index(drop=True)
        M = []
        P = []

        for i in range(t.shape[0]):
            # for i in range(150):
            id = t[['opt_date', 'PERMNO']].iloc[i, :]
            ind = (df['opt_date'] == id['opt_date']) & (df['PERMNO'] == id['PERMNO'])
            day = df.loc[ind, :]
            day.sort_values(['cp','strike'])
            m, p = self.pre_process_day(day, pred_col)

            M.append(m)
            P.append(p)
            if i % 100 == 0:
                print(i, '/', t.shape[0])

        iv_col = ['iv'+str(x) for x in np.arange(80,130,10)]
        if self.par.data.dtype in [DataType.OPTION_1,DataType.CRSP_OPTION_1, DataType.COMP_CRSP_OPTION_1]:
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
        m_df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/m_df.p')
        p_df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/pred_df.p')
        t.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/label_df.p')

        # debug
        # m_df = pd.read_pickle(self.par.data.dir+f'{self.par.data.dtype.name}/m_df.p')
        # p_df = pd.read_pickle(self.par.data.dir+f'{self.par.data.dtype.name}/pred_df.p')
        # t = pd.read_pickle(self.par.data.dir+f'{self.par.data.dtype.name}/label_df.p')

    def pre_process_day(self, day, pred_col):
        # create the filler function
        def fill_m_opt(x):
            return np.concatenate([x, np.zeros(Constant.MAX_OPT - len(x))])

        s0 = np.array(day.loc[:, 'S0'].iloc[0]).reshape(1)
        rf = np.array(day.loc[:, 'rf'].iloc[0]).reshape(1)

        t = day.loc[:, ['strike', 'opt_price']].copy().sort_values(['strike', 'opt_price']).reset_index(drop=True).groupby('strike').mean().reset_index()


        day.loc[day['cp'] == 'P', 'strike'].max()

        cb = CubicSpline(t['strike'], t['opt_price'])

        # K = np.arange(t['strike'].min(), t['strike'].max(), (t['strike'].max() - t['strike'].min()) / 200)
        K = np.linspace(t['strike'].min(), t['strike'].max(),200)
        assert len(K) == 200, 'Problem with the linespace, line 262'
        PRICE = cb(K)

        # plt.plot(t['strike'], t['opt_price'])
        # plt.scatter(t['strike'], t['opt_price'])
        # plt.plot(K, PRICE)
        # plt.show()

        t = day.loc[:, ['strike', 'impl_volatility']].copy().sort_values(['strike', 'impl_volatility']).reset_index(drop=True).groupby('strike').mean().reset_index()
        cb = CubicSpline(t['strike'], t['impl_volatility'])
        # X = np.arange(t['strike'].min(), t['strike'].max(), (t['strike'].max() - t['strike'].min()) / 200)
        X = s0 * np.arange(0.8, 1.3, 0.1)
        IV = cb(X)

        m = np.concatenate([K, PRICE, rf, s0])
        if self.par.data.dtype in [DataType.OPTION_1, DataType.CRSP_OPTION_1, DataType.COMP_CRSP_OPTION_1]:
            p = day.loc[:, pred_col].iloc[0, :].values
        p=np.concatenate([p,IV])

        return m, p

    def clean_opt_1(self):
        df = pd.read_csv(self.par.data.dir + 'raw/opt.csv')
        del df['issuer']
        del df['exercise_style']
        del df['index_flag']
        # del df['index_flag']
        df['cp_flag'] = df['cp_flag'].astype('category')
        df['ticker'] = df['ticker'].astype('category')
        print(df.head())
        print(df.dtypes)
        df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c1.p')

    def clean_opt_2(self):
        df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c1.p')
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # self.count_sample(df)

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

        df = df.loc[(df['T'] >= 28) & (df['T'] <= 31), :]
        # df = df.loc[(df['T'] == 28), :]

        df['day'] = df['date'].dt.dayofweek
        df['day'].hist(bins=25)
        plt.title('Day of week keep only T=28')
        plt.savefig('res/data/day_week_split_c1_t_clean.png')
        plt.close()

        ##################
        # strike and rename
        ##################
        df = df.rename(columns={'cp_flag': 'cp', 'strike_price': 'strike', 'brest_bid': 'bid', 'best_offer': 'ask', 'ticker': 'tic'})
        df['strike'] /= 1000

        ##################
        # Drop nan
        ##################
        df = df.dropna()
        df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c2.p')

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
        try:
            r['unique firm'] = df['tic'].unique().shape[0]
        except:
            r['unique firm'] = df['PERMNO'].unique().shape[0]

        try:
            r['unique day/firm'] = df[['date', 'tic']].drop_duplicates().shape[0]
        except:
            r['unique day/firm'] = df[['opt_date', 'tic']].drop_duplicates().shape[0]

        r['unique month/firm'] = df[['month', 'tic']].drop_duplicates().shape[0]

        if month_not_in:
            del df['month']

        return pd.Series(r, name=n)

    def clean_opt_3(self):
        df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c2.p')
        cleaner = [self.count_sample(df, 'Set T in [28,29,30,31]')]

        ##################
        # keep friday only
        ##################
        df['day'] = df['date'].dt.dayofweek

        # keep only friday to friday
        df = df.loc[df['day'] == 4, :]

        cleaner.append(self.count_sample(df, 'Keep only fridays'))

        ##################
        # keep only last friday
        ##################

        # keep only last friday of month
        df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
        t = df.groupby('month')['date'].transform('max')
        ind = df['date'] == t
        ind.mean()
        df = df.loc[ind, :]
        cleaner.append(self.count_sample(df, 'Keep last Friday'))

        ##################
        # check only one maturity per day
        ##################

        assert df.groupby(['tic', 'date'])['T'].std().max() == 0, 'Problem, multiple maturity on one day tic'

        ##################
        # remove wrong big and ask
        ##################
        ind = df['ask'] > 0
        df = df.loc[ind, :]
        ind = df['best_bid'] > 0
        df = df.loc[ind, :]

        ind = df['best_bid'] < df['ask']
        df = df.loc[ind, :]

        cleaner.append(self.count_sample(df, 'Set ask>0, bid>0 and bid<ask'))

        ##################
        # remove pure duplicates
        ##################

        ind = df[['strike', 'cp', 'T', 'date', 'ask', 'best_bid', 'secid']].duplicated()
        df = df.loc[~ind, :]

        cleaner.append(self.count_sample(df, 'Remove pure duplicated entries'))

        ##################
        # remove dupplicate in all but price
        ##################
        df['opt_price'] = (df['best_bid'] + df['ask']) / 2
        COL = ['cp', 'date', 'T', 'strike', 'secid']
        df = df.sort_values(COL + ['opt_price'])
        ind = df[COL].duplicated(keep='last')
        df = df.loc[~ind, :]

        cleaner.append(self.count_sample(df, 'Remove pure with all but price'))
        pd.DataFrame(cleaner)

        ##################
        # merge with price for cleaning
        ##################
        m = pd.read_csv(self.par.data.dir + 'raw/crsp_to_opt.csv')
        m['sdate'] = pd.to_datetime(m['sdate'], format='%Y%m%d')
        m['edate'] = pd.to_datetime(m['edate'], format='%Y%m%d')
        m.head()
        m = m.sort_values('PERMNO')
        for i in range(1, 6):
            df[f'PERMNO{i}'] = -1
        df['PERMNO'] = np.nan
        for secid in df['secid'].unique():
            temp = m.loc[m['secid'] == secid, :]

            for permno in temp['PERMNO'].unique():
                score = temp.loc[temp['secid'] == secid, 'score'].values[0]
                sdate = temp.loc[temp['secid'] == secid, 'sdate'].values[0]
                edate = temp.loc[temp['secid'] == secid, 'edate'].values[0]
                ind = (df['secid'] == secid) & (df['date'] >= sdate) & (df['date'] <= edate)
                df.loc[ind, f'PERMNO{score}'] = permno

        for i in range(1, 6):
            ind = df[f'PERMNO{i}'] > -1
            df.loc[ind, 'PERMNO'] = df.loc[ind, f'PERMNO{i}']
        for i in range(1, 6):
            del df[f'PERMNO{i}']

        raw = RawData(par=self.par, freq='daily')
        raw.crsp.head()
        df.head()
        df = df.merge(raw.crsp[['date', 'PERMNO', 'price']])

        cleaner.append(self.count_sample(df, 'Merge with CRSP price'))

        ##################
        # drop in the money options
        ##################
        df.head()
        ind = (df['cp'] == 'C') & (df['strike'] < df['price'])
        df = df.loc[~ind, :]
        ind = (df['cp'] == 'P') & (df['strike'] > df['price'])
        df = df.loc[~ind, :]

        cleaner.append(self.count_sample(df, 'Drop OTM'))

        #################
        # remove  if less than 15 different option strike price
        ##################

        t = df.groupby(['date', 'secid'])['strike'].nunique()
        df = df.merge(t.reset_index().rename(columns={'strike': 'nb_distinct'}))
        ind = df['nb_distinct'] >= 15
        print('Drop because less than 5 strike', 1 - ind.mean())
        df = df.loc[ind, :]
        del df['nb_distinct']

        cleaner.append(self.count_sample(df, 'Set minimum 15 diff. strikes per date/tic'))

        ##################
        # remove if volume is too low
        ##################

        # days with rule --> at least 84.5 different options per maturity
        t = df.groupby(['date', 'secid'])['volume'].sum()
        df = df.merge(t.reset_index().rename(columns={'volume': 'vol_tot'}))
        ind = df['vol_tot'] >= 20
        print('Drop volume total below 20', 1 - ind.mean())
        df = df.loc[ind, :]
        del df['vol_tot']

        cleaner.append(self.count_sample(df, 'Set min sum(volume)>=20'))

        ##################
        # remove strike price ratio
        ##################

        # df.head()
        # df['sp_ratio'] = df['strike']/df['price']
        # 1-0.2*np.sqrt(1/12)
        # t = df.groupby(['tic','date'])['sp_ratio'].agg(['min','max'])
        # df = df.merge(t.reset_index())
        # ind=(df['min']<1-0.2*np.sqrt(1/12)) & (df['max']>1+0.2*np.sqrt(1/12))
        # print('Drop because strike price ratio', 1-ind.mean())
        # df = df.loc[ind,:]
        #
        # del df['min'], df['max'], df['sp_ratio']
        #
        # cleaner.append(self.count_sample(df, 'Remove days with strike price ratio'))

        # ##################
        # # remove day option when maximum diff between neighbooring strike price is greater than max of 20% o the closing stock price
        # ##################
        # df=df.sort_values(['date','tic','cp','strike'])
        # df.head()
        # t=df.groupby(['date','tic','cp'])['strike'].diff()
        # df['d'] = t.values
        # t=df.groupby(['date','tic'])['d'].max()
        #
        # df = df.merge(t.reset_index().rename(columns={'d': 'd_max'}))
        # ind=(df['d_max']<= df['price']*0.2) | (df['d_max']<=10)
        # print('Drop adjascent price are too close', 1-ind.mean())
        # df = df.loc[ind,:]
        #
        # cleaner.append(self.count_sample(df, 'Remove days with diff. between strike too large'))

        df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c3.p')
        pd.DataFrame(cleaner).to_pickle(self.par.data.dir + 'side/cleaner_c3.p')

    def clean_opt_4(self):
        df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_c3.p')
        cleaner = pd.read_pickle(self.par.data.dir + 'side/cleaner_c3.p')
        del df['day']
        df['month'] = df['date'].dt.year * 100 + df['date'].dt.month

        raw = RawData(self.par, 'daily')
        raw.crsp = raw.crsp.rename(columns={'price': 'S0'})
        raw.crsp = raw.crsp.sort_values(['PERMNO', 'date']).reset_index(drop=True)

        T = 28
        raw.crsp['S_T'] = raw.crsp.groupby(['PERMNO'])['S0'].shift(-T)
        raw.crsp['adj_T'] = raw.crsp.groupby(['PERMNO'])['adj'].shift(-T)
        raw.crsp['S_T'] = raw.crsp['S_T'] * raw.crsp['adj_T'] / raw.crsp['adj']
        raw.crsp['ret'] = (raw.crsp['S_T'] / raw.crsp['S0']) - 1

        df = df.merge(raw.crsp[['date', 'PERMNO', 'S0', 'S_T', 'ret']])
        df = df.rename(columns={'date': 'opt_date'})

        cleaner = cleaner.append(self.count_sample(df, 'Computed returns'))

        df.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_final.p')
        print('#' * 10, 'final cleaner')
        print(cleaner)
        cleaner.to_pickle(self.par.data.dir + 'side/cleaner_final.p')

    def load_opt(self):
        df = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/opt_final.p')

        return df

    def load_pred_crsp_only(self, reload=False):
        if reload:
            ##################
            # recreating the appendix C variable of the Kadan Tang paper (RFS)
            ##################
            var_list = ['mkt_cap', 'mom_f']
            ################# The monthly ones
            raw = RawData(self.par, freq='monthly')
            df = raw.crsp
            ### mkt cap
            df['mkt_cap'] = np.log(1000 * df['share'] * df['price'])
            ### momentum
            df = df.sort_values(['PERMNO', 'date']).reset_index(drop=True)
            df['r'] = np.log(df['ret'] + 1)
            df.index = df['date']
            t = df.groupby('PERMNO')['r'].rolling(12).sum().reset_index().rename(columns={'r': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)
            final = df[['PERMNO', 'date', 'crsp_date'] + var_list].copy()
            ################ The daily ones
            #### amihud's liquidity group per month
            raw = RawData(self.par, freq='daily')
            df = raw.crsp.copy()
            df['liq_m'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby(['PERMNO', 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby('PERMNO')['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final = final.merge(t)
            #### amihud's liquidity group per year
            print(final.shape)
            df = raw.crsp.copy().sort_values(['PERMNO', 'date'])
            df['liq_y'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby(['PERMNO', 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby('PERMNO')['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')
            var_list.append('liq_m')
            var_list.append('liq_y')
            ##################
            # adding the one got through regression
            ##################
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
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
            final.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/pred_crsp.p')
        else:
            final = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/pred_crsp.p')
        return final

    def load_pred_compustat_and_crsp(self, reload=False):
        if reload:
            ##################
            # recreating the appendix C variable of the Kadan Tang paper (RFS)
            ##################
            var_list = ['mkt_cap', 'btm', 'mom_f', 'inv', 'prof']
            ################# The monthly ones
            raw = RawData(self.par, freq='monthly')
            df = raw.crsp.copy()
            ### mkt cap
            df['mkt_cap'] = np.log(1000 * df['share'] * df['price'])
            ### Book to market
            df['year'] = df['date'].dt.year
            comp = raw.compustat.loc[raw.compustat['quarter'] == 4, ['year', 'PERMNO', 'asset', 'debt']].sort_values(['PERMNO', 'year']).reset_index(drop=True)
            comp['asset_l'] = comp.groupby('PERMNO')['asset'].shift(1)
            df = df.merge(comp)
            df['btm'] = 1000 * 1000 * (df['asset'] - df['debt']) / (df['share'] * df['price'])
            ### momentum
            df = df.sort_values(['PERMNO', 'date']).reset_index(drop=True)
            df['r'] = np.log(df['ret'] + 1)
            df.index = df['date']
            t = df.groupby('PERMNO')['r'].rolling(12).sum().reset_index().rename(columns={'r': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)
            ### investment
            df['inv'] = df['asset'] - df['asset_l']
            ### Profitability
            comp = raw.compustat.loc[:, ['date', 'PERMNO', 'asset', 'debt']].sort_values(['PERMNO', 'date']).reset_index(drop=True)
            comp['book_equity'] = comp['asset'] - comp['debt']
            comp['book_equity'] = comp.groupby('PERMNO')['book_equity'].shift(1)
            df = df.merge(comp, how='left')
            df.index = df['date']
            t = df.groupby('PERMNO').apply(lambda x: x['book_equity'].fillna(method='ffill')).reset_index().drop_duplicates()
            df = df.reset_index(drop=True).drop(columns='book_equity')
            df = df.merge(t, how='left')
            comp = raw.compustat_yearly.drop_duplicates()
            df = df.merge(comp)
            df['prof'] = df['income'] / df['book_equity']
            final = df[['PERMNO', 'date', 'crsp_date'] + var_list].copy()
            ################ The daily ones
            #### amihud's liquidity group per month
            raw = RawData(self.par, freq='daily')
            df = raw.crsp.copy()
            df['liq_m'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['date'] = df['date'].dt.year * 100 + df['date'].dt.month
            t = df.groupby(['PERMNO', 'date'])['liq_m'].mean().reset_index()
            t['liq_m'] /= t['liq_m'].mean()
            t['date'] = pd.to_datetime(t['date'], format='%Y%m')
            t.index = t['date']
            t['liq_m'] = t.groupby('PERMNO')['liq_m'].shift(-1)
            t = t.reset_index(drop=True)
            final = final.merge(t)
            #### amihud's liquidity group per year
            print(final.shape)
            df = raw.crsp.copy().sort_values(['PERMNO', 'date'])
            df['liq_y'] = df['ret'].abs() / df['vol']
            df = df.loc[df['vol'] > 0, :]
            df['year'] = df['date'].dt.year
            t = df.groupby(['PERMNO', 'year'])['liq_y'].mean().reset_index()
            t['liq_y'] /= t['liq_y'].mean()
            t.index = t['year']
            t['liq_y'] = t.groupby('PERMNO')['liq_y'].shift(-1)
            t = t.reset_index(drop=True)
            final['year'] = final['date'].dt.year
            final = final.merge(t).drop(columns='year')
            var_list.append('liq_m')
            var_list.append('liq_y')
            ##################
            # adding the one got through regression
            ##################
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_m.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + f'int_gen/beta_daily.p')
            final = final.merge(df)
            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_m.p')
            final = final.merge(df)
            # check this one
            df = pd.read_pickle(self.par.data.dir + f'int_gen/id_risk_daily.p')
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
            final.to_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/pred_crsp_compustat.p')
        else:
            final = pd.read_pickle(self.par.data.dir + f'{self.par.data.dtype.name}/int/pred_crsp_compustat.p')
        return final

    def get_beta(self, freq='monthly'):
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
            if x.shape[0] > r / 4:
                return sm.OLS(x['ret'] - x['rf'], x[['one', 'mkt-rf']]).fit().params['mkt-rf']
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('PERMNO').apply(get_delta)
            t.name = date
            res.append(t)
            date += pd.DateOffset(months=1)
            date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.month == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'PERMNO', 'value': f'beta_{freq}'})
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
            r = 60
        else:
            r = 12

        raw = RawData(self.par, freq)
        df = raw.ff.merge(raw.crsp, how='inner', on='date')
        df['one'] = 1
        df = df.dropna()

        def get_std(x):
            if x.shape[0] > r / 4:
                return np.std(sm.OLS(x['ret'] - x['rf'], x[['one', 'mkt-rf']]).fit().resid)
            else:
                return np.nan

        date = df['date'].min() + pd.DateOffset(months=r)
        date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')

        date_end = df['date'].max()

        res = []
        while date <= date_end:
            ind = (df['date'] < date) & (df['date'] >= date + pd.DateOffset(months=-r))
            t = df.loc[ind, :].groupby('PERMNO').apply(get_std)
            t.name = date
            res.append(t)
            date += pd.DateOffset(months=1)
            date = pd.to_datetime(date.year * 100 + date.month, format='%Y%m')
            if date.month == 1:
                print(date)

        t = pd.DataFrame(res)
        t.index.name = 'date'
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': 'PERMNO', 'value': f'id_risk_{freq}'})
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
        self.clean_opt_1()
        self.clean_opt_2()
        self.clean_opt_3()
        self.clean_opt_4()

        if self.par.data.dtype in [DataType.CRSP_1,DataType.CRSP_OPTION_1]:
            self.load_pred_crsp_only(reload=True)
        if self.par.data.dtype in [DataType.COMP_CRSP_OPTION_1,DataType.COMP_CRSP_1]:
            self.load_pred_compustat_and_crsp(reload=True)

        self.create_a_dataset()

par = Params()
# par.data.dtype = DataType.CRSP_OPTION_1
self = Data(par)
# self.load_final()
# self.move_shuffle()
# self.test_m_df
