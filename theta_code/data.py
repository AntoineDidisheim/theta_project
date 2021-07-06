import pandas as pd
import numpy as np
from parameters import *
import os
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import socket

if socket.gethostname() != 'work':
    import matplotlib

    matplotlib.use('Agg')
from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from Econ import Econ
from scipy import optimize as opti
from scipy import stats
from multiprocessing import Pool


# from pandarallel import pandarallel


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
        self.id_col = 'permno'

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

        print(f'Set training year {year}', flush=True)

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
        print(f'Set shuffle {self.shuffle_id}', flush=True)

    def load_final(self):
        # finally load all all
        if os.path.isfile(self.par.data.dir + f'{self.name}/m_df.p'):
            self.m_df = pd.read_pickle(self.par.data.dir + f'{self.name}/m_df.p').reset_index(drop=True)
            self.p_df = pd.read_pickle(self.par.data.dir + f'{self.name}/pred_df.p').reset_index(drop=True)
            self.label_df = pd.read_pickle(self.par.data.dir + f'{self.name}/label_df.p').reset_index(drop=True)
        else:
            d = self.par.data.dir + f'{self.name}/temp_m/'
            D = []
            for f in os.listdir(d):
                D.append(int(f.split('.p')[0].split('_')[-1]))
            D = np.unique(D)
            self.m_df = pd.DataFrame()
            self.p_df = pd.DataFrame()
            self.label_df = pd.DataFrame()
            for i in D:
                print('load batch ', i)
                self.m_df = self.m_df.append(
                    pd.read_pickle(d + f'm_df_{i}.p')
                )
                self.label_df = self.label_df.append(
                    pd.read_pickle(d + f'label_df_{i}.p')
                )
                self.p_df = self.p_df.append(
                    pd.read_pickle(d + f'pred_df_{i}.p')
                )

                # print(pd.read_pickle(d+f'label_df_{i}.p'))
                print('#' * 50, i)
                print(pd.read_pickle(d + f'label_df_{i}.p').iloc[:, :5])

        if self.par.data.hist_theta:
            hist = self.historical_theta()
            t = self.label_df.merge(hist, how='left')['theta']
            self.p_df['theta_hist'] = t

        if self.par.data.vilk_theta:
            t = pd.read_pickle(self.par.data.dir+'raw_merge/v_theta.p').drop_duplicates()
            self.label_df=self.label_df.merge(t, how='left')

            self.label_df=self.label_df.merge(t,how='left')
            self.label_df['theta_v']=self.label_df['theta_v'].fillna(self.label_df['theta_v'].mean())
            self.p_df['theta_v'] = self.label_df['theta_v'].values

        if self.par.data.var_subset is not None:
            self.p_df=self.p_df.loc[:,self.par.data.var_subset]



        # add the transofrmed return
        self.label_df['log_ret'] = self.label_df['ret']
        self.label_df['normal_ret'] = np.exp(self.label_df['log_ret']) - 1

        if self.par.data.ret == ReturnType.RET:
            self.label_df['ret'] = self.label_df['normal_ret']
        if self.par.data.ret == ReturnType.LOG:
            self.label_df['ret'] = self.label_df['log_ret']

        if self.par.data.max_ret > -1:
            ind = (self.label_df['ret'] >= self.par.data.min_ret) & (self.label_df['ret'] <= self.par.data.max_ret)
            self.m_df = self.m_df.loc[ind, :].reset_index(drop=True)
            self.p_df = self.p_df.loc[ind, :].reset_index(drop=True)
            self.label_df = self.label_df.loc[ind, :].reset_index(drop=True)



        # deal with remaining inf
        self.p_df = self.p_df.replace({-np.inf: np.nan, np.inf: np.nan})

        for c in self.p_df.columns:
            self.p_df.loc[:, c] = (self.p_df[c] - self.p_df[c].mean()) / self.p_df[c].std()

        self.p_df = self.p_df.fillna(0)
        # self.p_df.quantile(0.999)
        self.p_df = self.p_df.clip(-3.0, 3.0)

    def load_and_merge_pred_opt(self, reload=False):
        # df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/opt.p')
        if reload:
            cleaner = pd.read_pickle(self.par.data.dir + f'{self.name}/side/cleaner_all.p')
            print('a')
            opt = self.load_opt()
            print(opt)
            print('b')
            if self.par.data.crsp:
                if self.par.data.comp:
                    pred = self.load_pred_compustat_and_crsp()
                else:
                    pred = self.load_pred_crsp_only()
            print('c')

            # pred['month'] = pred['date'].dt.year * 100 + pred['date'].dt.month
            # del pred['date']

            df = pred.merge(opt, on=[self.id_col, 'date'])
            print('d')
            # cleaner = cleaner.append(self.count_sample(df, 'Merge with predictors'))
            print('e')

            df = df.dropna()
            print('f')
            cleaner = cleaner.append(self.count_sample(df, 'Drop missing predictors'))
            print('g')
            print(cleaner, flush=True)

            ind = df['cp'] == 'C'
            mc = df.loc[ind, :].groupby(['date', self.id_col])['ret'].count().max()
            ind = df['cp'] == 'P'
            mp_ = df.loc[ind, :].groupby(['date', self.id_col])['ret'].count().max()
            print('Max nb call/put in sample:', mc, mp_, flush=True)
            cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_after_merge.csv')

            df.to_pickle(self.par.data.dir + f'{self.name}/int/opt_merge.p')

        else:
            df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/opt_merge.p')
        return df

    def create_a_dataset_batch(self, batch_id=0, batch_size=25000):
        print('In create dataset', flush=True)

        if self.par.data.crsp:
            if self.par.data.comp:
                pred = self.load_pred_compustat_and_crsp()
            else:
                pred = self.load_pred_crsp_only()
        print('Load pred', flush=True)

        # specify here the list of DataType which do not need to be merge with any dataset
        if self.par.data.opt and (not self.par.data.comp) and (not self.par.data.crsp):
            df = self.load_opt()
            raw = RawData(self.par, 'daily')
            df = df.merge(raw.ff[['date', 'rf']])
            pred_col = []
        else:
            print('start load merge and pred')
            df = self.load_and_merge_pred_opt(True)
            print('start next step')
            pred_col = list(pred.drop(columns=[self.id_col, 'date']).columns)

        print('Loaded df', flush=True)

        if self.par.data.mw:

            if self.id_col == 'permno':
                mw = pd.read_csv(self.par.data.dir + 'bench/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
                mw['date'] = pd.to_datetime(mw['date'])
                mw['permno'] = mw['permno'].astype(int)
                mw['MW'] = mw['mw30'] / 12
                mw = mw[['date', self.id_col, 'MW']]
            else:
                mw = self.marting_wagner_return()

            gl = pd.read_csv(self.par.data.dir + 'bench/glb_daily.csv')
            gl['date'] = pd.to_datetime(gl['date'])
            gl = gl[['date', 'id', 'glb3_D30']]
            gl.columns = ['date', self.id_col, 'them']
            mw = mw.merge(gl)

            ind = mw[['date', self.id_col]].duplicated(keep='first')
            mw = mw.loc[~ind, :]
            df = df.merge(mw, how='left')
            pred_col.append('MW')
            pred_col.append('them')

        print('Loading finish', flush=True)

        ##################
        # process a day
        ##################
        # select a day
        t = df[['date', self.id_col, 'ret']].drop_duplicates().reset_index(drop=True)
        M = []
        P = []

        if 'rf_x' in df.columns:
            df['rf'] = df['rf_x'].values
            del df['rf_x']

        RF = self.load_rf()

        print('Start loop', flush=True)
        ERR = []

        i_start = batch_id * batch_size
        i_end = min(t.shape[0], (batch_id + 1) * batch_size)
        t_ind_keep = []
        if i_end < t.shape[0]:
            for i in range(i_start, i_end):
                # for i in range(i_start,i_start+10):
                # for i in range(100):
                id = t[['date', self.id_col]].iloc[i, :]
                ind = (df['date'] == id['date']) & (df[self.id_col] == id[self.id_col])
                day = df.loc[ind, :]
                day = day.loc[day['delta'].abs() <= 0.5, :]

                if day.loc[:, ['strike', 'opt_price', 'impl_volatility']].drop_duplicates().shape[0] > 1:
                    try:
                        m, p = self.pre_process_day(day, pred_col, RF)

                        M.append(m)
                        P.append(p)
                        t_ind_keep.append(i)
                    except:
                        ERR.append(i)
                else:
                    print(f'### Skip {i}, not enough points')

                if i % 100 == 0:
                    print(i, '/', t.shape[0], flush=True)

            print('finished')
            print('Nb err:', len(ERR))
            print(ERR)

            # select only the right t, to save the minibatch correctly
            t = t.loc[t_ind_keep, :]

            ### end apply

            iv_col = ['iv' + str(x) for x in np.arange(80, 130, 10)]
            if self.par.data.opt:
                pred_col = pred_col + iv_col
            m_df = pd.DataFrame(M)
            p_df = pd.DataFrame(P, columns=pred_col)

            # find na in any df

            m_df = m_df.reset_index(drop=True)
            p_df = p_df.reset_index(drop=True)
            t = t.reset_index(drop=True)

            temp = pd.DataFrame(np.concatenate([t, m_df, p_df], 1))
            ind = pd.isna(temp).sum(1) == 0

            m_df = m_df.loc[ind, :].reset_index(drop=True)
            p_df = p_df.loc[ind, :].reset_index(drop=True)
            t = t.loc[ind, :].reset_index(drop=True)

            os.makedirs(self.par.data.dir + f'{self.name}/temp_m/', exist_ok=True)
            # finally save all
            m_df.to_pickle(self.par.data.dir + f'{self.name}/temp_m/m_df_{batch_id}.p')
            p_df.to_pickle(self.par.data.dir + f'{self.name}/temp_m/pred_df_{batch_id}.p')
            t.to_pickle(self.par.data.dir + f'{self.name}/temp_m/label_df_{batch_id}.p')

    def create_a_dataset(self):
        print('In create dataset', flush=True)

        if self.par.data.crsp:
            if self.par.data.comp:
                pred = self.load_pred_compustat_and_crsp()
            else:
                pred = self.load_pred_crsp_only()
        print('Load pred', flush=True)

        # specify here the list of DataType which do not need to be merge with any dataset
        if self.par.data.opt and (not self.par.data.comp) and (not self.par.data.crsp):
            df = self.load_opt()
            raw = RawData(self.par, 'daily')
            df = df.merge(raw.ff[['date', 'rf']])
            pred_col = []
        else:
            print('start load merge and pred')
            df = self.load_and_merge_pred_opt(True)
            print('start next step')
            pred_col = list(pred.drop(columns=[self.id_col, 'date']).columns)

        print('Loaded df', flush=True)

        if self.par.data.mw:
            mw = self.marting_wagner_return()

            ind = mw[['date', self.id_col]].duplicated(keep='first')
            mw = mw.loc[~ind, :]
            df = df.merge(mw, how='left')
            pred_col.append('MW')

        print('Loading finish', flush=True)

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
        print('Start loop', flush=True)
        ERR = []
        for i in range(t.shape[0]):
            # for i in range(100):
            id = t[['date', self.id_col]].iloc[i, :]
            ind = (df['date'] == id['date']) & (df[self.id_col] == id[self.id_col])
            day = df.loc[ind, :]
            day = day.loc[day['delta'].abs() <= 0.5, :]

            if day.loc[:, ['strike', 'opt_price', 'impl_volatility']].drop_duplicates().shape[0] > 1:
                try:
                    m, p = self.pre_process_day(day, pred_col, RF)

                    M.append(m)
                    P.append(p)
                except:
                    ERR.append(i)
            else:
                print(f'### Skip {i}, not enough points')

            if i % 100 == 0:
                print(i, '/', t.shape[0], flush=True)

        print('finished')
        print('Nb err:', len(ERR))
        print(ERR)
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

    def pre_process_day(self, day, pred_col, RF):

        s0 = np.array(day.loc[:, 'S0'].iloc[0]).reshape(1)
        try:
            rf = RF.loc[RF.iloc[:, 0] == day['date'].iloc[0], :].iloc[0, 1]
        except:
            rf = np.array(day.loc[:, 'rf'].iloc[0]).reshape(1)
            print('missing rf', flush=True)

        t = day.loc[:, ['strike', 'opt_price', 'impl_volatility']].copy().sort_values(['strike', 'opt_price', 'impl_volatility']).reset_index(drop=True).groupby('strike').mean().reset_index()

        # cb = CubicSpline(t['strike'], t['opt_price'])
        bound = (t['impl_volatility'].iloc[0], t['impl_volatility'].iloc[-1])

        if self.par.data.opt_smooth in [OptSmooth.EXT_CUBIC, OptSmooth.VOLA_CUBIC]:
            cb = CubicSpline(t['strike'], t['impl_volatility'], extrapolate=False)
        else:
            cb = interp1d(t['strike'], t['impl_volatility'], bounds_error=False, fill_value=bound)

        if self.par.data.opt_smooth in [OptSmooth.EXT, OptSmooth.EXT_CUBIC, OptSmooth.VOLA_CUBIC]:
            K = np.linspace(s0 * 1 / 3, s0 * 3, Constant.GRID_SIZE)
        if self.par.data.opt_smooth == OptSmooth.INT:
            K = np.linspace(t['strike'].min(), t['strike'].max(), Constant.GRID_SIZE)
        assert len(K) == Constant.GRID_SIZE, 'Problem with the linespace'

        IV = cb(K)
        if np.any(pd.isna(IV)):
            IV[0] = IV[~np.isnan(IV)][0]
            IV = pd.DataFrame(IV).fillna(method='ffill').values

        def BlackScholes_price(S, r, sigma, K):
            dt = 28 / 365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S > K] = pr_put[S > K]
            return pr

        S = day['S'].iloc[0]

        PRICE = BlackScholes_price(S, rf, IV, K)

        # plt.plot(K,IV)
        # # plt.xlim(39,55)
        # plt.show()
        #
        # plt.plot(K, PRICE)
        # # plt.xlim(39, 55)
        # plt.show()

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

        if self.par.data.opt_smooth in [OptSmooth.EXT, OptSmooth.EXT_CUBIC, OptSmooth.VOLA_CUBIC]:
            m = np.concatenate([K[:, 0], PRICE[:, 0], [rf], s0])
        if self.par.data.opt_smooth == OptSmooth.INT:
            m = np.concatenate([K, PRICE, [rf], s0])
        p = day.loc[:, pred_col].iloc[0, :].values

        if self.par.data.opt:
            p = np.concatenate([p, IV])

        return m, p

    def load_rf(self, reload=False):
        if reload:
            rf = pd.read_csv(f'{self.par.data.dir}raw/rf.csv')
            rf['date'] = pd.to_datetime(rf['date'], format='%Y%m%d')
            rf['one'] = 1
            prb_date = []
            df = pd.DataFrame()
            for d in rf['date'].unique():
                # temp_df = df.loc[df['date'] == d, :]
                temp = rf.loc[(rf['date'] == d), :]
                # tr_days = temp.loc[(temp['days']<=temp_df['T'].max()),'days']
                # tr_rf = temp.loc[(temp['days']<=temp_df['T'].max()),'rate']
                if temp.shape[0] > 0:
                    m = KNeighborsRegressor(n_neighbors=2, weights='distance').fit(temp[['one', 'days']], temp['rate'])
                    t = pd.DataFrame(data={'one': [1.0], 'days': [30]})
                    df = df.append(
                        pd.DataFrame(data={'date': [d], 'rf': m.predict(t) / 100})
                    )
                    #
                    # t=np.linspace(temp['days'].min(),365*2,200)
                    # t = pd.DataFrame(data={'one':np.ones_like(t),'days':t})
                    # t['rate'] = m.predict(t)
                    # plt.plot(t['days'],t['rate'])
                    # ind = temp['days']<=365*2
                    # plt.scatter(temp.loc[ind,'days'],temp.loc[ind,'rate'])
                    # plt.show()
                else:
                    print('problem: ', d)
                    prb_date.append(d)

            df['rf'] = ((1 + df['rf']) ** (1 / 12)) - 1

            dates = []
            d = df['date'].min()
            while d < df['date'].max():
                dates.append(d)
                d += pd.DateOffset(days=1)
            df = df.merge(pd.DataFrame({'date': dates}), how='outer')
            df['rf'] = df['rf'].fillna(method='ffill')
            df['rf'] = df['rf'].fillna(method='bfill')

            df.to_pickle(self.par.data.dir + 'raw/rf.p')
            print('RF recomputed', df['rf'].min(), df['rf'].max())
        else:
            df = pd.read_pickle(self.par.data.dir + 'raw/rf.p')
        return df

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
            print('index name and ticker already del', flush=True)

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

        t = self.load_all_price()[[self.id_col, 'date', 'S0', 'ret']]

        # transform gvkey to correct format
        df[self.id_col] = df[self.id_col].apply(lambda x: str(x)[:6])
        df[self.id_col] = pd.to_numeric(df[self.id_col], errors='coerce')
        df = df.loc[~pd.isna(df[self.id_col]), :]
        df[self.id_col] = df[self.id_col].astype(int)

        df = df.merge(t, on=[self.id_col, 'date'])
        cleaner = cleaner.append(self.count_sample(df, 'Computed returns'))

        print('#' * 10, 'final cleaner of year', year, '#' * 10)
        print(cleaner, flush=True)
        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_{year}.p')
        return df, cleaner

    def clean_vola_surface(self):

        df = pd.read_csv(self.par.data.dir + 'raw/opt_surf.csv')
        tr = pd.read_csv('data/crsp_to_metrics.csv')
        tr.columns = [x.lower() for x in tr.columns]
        tr = tr[['permno', 'ticker']].dropna()
        t = tr.groupby('ticker').transform('nunique')
        tr = tr.loc[t.values.flatten() == 1, :].drop_duplicates()

        df = df.merge(tr)
        df.groupby('date')['ticker'].nunique()

        df = df.rename(columns={'days': 'T', 'cp_flag': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares', 'impl_strike': 'strike',
                                's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'impl_premium': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                'expiration': 'exdate'})
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        df = df.loc[df['date'].dt.dayofweek == 4, :]  ## keep only fridays

        df['exdate'] = df['date'] + pd.DateOffset(days=df['T'].iloc[0])
        df['opt_price'] = df['o_ask']
        df['option_id'] = df.index

        df['delta'] /= 100
        df['permno'] = df['permno'].astype(int)
        # price = pd.read_csv(self.par.data.dir + 'raw/spx_price.csv').rename(columns={'close': 'S'})[['date', 'S']]
        # price['date'] = pd.to_datetime(price['date'], format='%Y%m%d')
        price = self.load_all_price(True)
        print(df.shape)
        df = df.merge(price)
        print(df.shape)

        ##################
        # clean option
        ##################

        df = df.rename(columns={'t': 'T', 'call_put': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares',
                                's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                'expiration': 'exdate'})
        L = df.columns
        for c in L:
            if c not in ['date', self.id_col, 'T', 'impl_volatility', 'delta', 'strike', 'opt_price', 'S', 'cp', 'ticker', 'ret']:
                del df[c]

        df.head()
        df['S0'] = df['S']

        df['cp'] = df['cp'].astype('str')

        ##################
        # strike
        ##################

        cleaner = pd.DataFrame([self.count_sample(df, 'Raw')])
        ##################
        # Drop nan
        ##################

        df = df.dropna()

        cleaner = cleaner.append(self.count_sample(df, 'Drop na'))

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
        # save output
        ##################

        cleaner.to_pickle(self.par.data.dir + f'{self.name}/side/cleaner_all.p')
        print(cleaner, flush=True)
        df.to_pickle(self.par.data.dir + f'{self.name}/int/opt.p')

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
        c = pd.read_pickle(self.par.data.dir + f'{self.name}/side/cleaner_all.p')

        df.to_pickle(self.par.data.dir + f'{self.name}/int/opt.p')

    def load_all_price(self, reload=False):
        if reload:
            # Minimum columns: ['PERMNO','CUSIP','TICKER', 'date', 'PRC', 'RET', 'CFACPR']
            df = pd.read_csv(self.par.data.dir + '/raw/crsp_4.csv')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df.columns = [x.lower() for x in df.columns]
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df = df.loc[~pd.isna(df['ret']), :]
            df = df.sort_values(['permno', 'date']).reset_index(drop=True)
            ## We compute here the returns for various horizon h
            # in crsp returns at time t are buying t-1, selling t
            # here we do buying t, selling t+h
            ## remove the debug below to visualize
            # Debug df for visualisation
            # df = pd.DataFrame({'prc':[1,2,3,4,5,6,7,8,9,10]})
            # df['ret']=(df['prc']/df['prc'].shift(1)-1)
            df['log_ret'] = np.log(df['ret'] + 1)
            H = [1, 20]
            ret_col = []
            H_col = []
            for h in H:
                h_name = h
                if h % 20 == 0:
                    h_name = f'{int(h / 20)}m'
                if h % 252 == 0:
                    h_name = f'{int(h / 252)}y'
                ret_col.append(f'ret{h_name}')
                H_col.append(f'H{h_name}')
                df[f'ret{h_name}'] = df.groupby('permno')['log_ret'].rolling(h).sum().shift(-h).reset_index()['log_ret']
                df[f'ret{h_name}'] = np.exp(df[f'ret{h_name}']) - 1
                t = df.groupby('permno')['date'].shift(-h).reset_index()['date']
                tt = (t - df['date']).dt.days
                # remove random skip in days
                df.loc[tt > tt.quantile(0.99), f'ret{h_name}'] = np.nan
                df[f'H{h_name}'] = tt

            df = df[['permno', 'ticker', 'date', 'shrout', 'bid', 'ask', 'vol'] + ret_col].rename(columns={'shrout': 'shares_outstanding', 'ret1': 'ret_1', 'ret1m': 'ret', 'vol': 'total_volume'})
            df['S'] = (df['bid'] + df['ask']) / 2
            df['S0'] = (df['bid'] + df['ask']) / 2

            df.to_pickle(self.par.data.dir + f'{self.name}/int/price.p')
        else:
            df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/price.p')
        return df

    def load_all_price_old(self, reload=False):
        if reload:
            if self.par.data.opt_smooth in [OptSmooth.VOLA_CUBIC]:
                df = pd.read_csv(self.par.data.dir + 'raw/' + 'price_surf_2.csv')
                df.columns = [x.lower() for x in df.columns]
                df['ret']=pd.to_numeric(df['ret'],errors='coerce').fillna(0.0)
                df['log_ret'] = np.log(df['ret'] + 1)

                df['S0'] = (df['ask'] + df['bid']) / 2
                df['S'] = (df['ask'] + df['bid']) / 2
                df = df.rename(columns={'gv_key': 'gvkey', 'cfacpr': 'adj', 'vol': 'total_volume', 'shrout': 'shares_outstanding'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df
                df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)
                df['S0'] = df['S0'].abs()

                T = 1
                df[f'ret_1'] = df.groupby(self.id_col)['log_ret'].rolling(T).sum().shift(-T).reset_index()['log_ret']
                # df[f'h{h}'] = np.exp(df[f'h{h}']) - 1
                # df['S_T'] = df.groupby([self.id_col])['S0'].shift(-T)
                # df['adj_T'] = df.groupby([self.id_col])['adj'].shift(-T)
                # df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
                # df['ret_1'] = np.log(df['S_T'] / df['S0'])

                T = 20

                df['S_T'] = df.groupby([self.id_col])['S0'].shift(-T)
                df['adj_T'] = df.groupby([self.id_col])['adj'].shift(-T)
                df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
                # df['ret'] = np.log(df['S_T'] / df['S0'])
                df[f'ret'] = df.groupby(self.id_col)['log_ret'].rolling(T).sum().shift(-T).reset_index()['log_ret']

                df[self.id_col] = pd.to_numeric(df[self.id_col], errors='coerce')
                df = df.loc[~pd.isna(df[self.id_col]), :]
                df[self.id_col] = df[self.id_col].astype(int)

                df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)

                df.to_pickle(self.par.data.dir + f'{self.name}/int/price.p')
            else:

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

                if self.par.data.opt_smooth in [OptSmooth.VOLA_CUBIC]:
                    T = 30
                else:
                    T = 28
                df['S_T'] = df.groupby([self.id_col])['S0'].shift(-T)
                df['adj_T'] = df.groupby([self.id_col])['adj'].shift(-T)
                df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
                df['ret'] = np.log(df['S_T'] / df['S0'])
                # df=df.dropna()
                if self.id_col == 'gvkey':
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

        tr = pd.read_csv('data/crsp_comp.csv')[['GVKEY', 'LPERMNO', 'fyearq']]
        tr.columns = ['gvkey', 'permno', 'year']
        df['year'] = df['date'].dt.year

        t = tr.merge(df[['gvkey', 'year']].drop_duplicates())
        tt = t.groupby(['gvkey', 'year']).transform('nunique').values
        t = t.loc[tt.flatten() == 1, :]
        df = df.merge(t)

        del df['year']
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
            print(final.shape, flush=True)
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
            df = df.sort_values([self.id_col, 'date']).reset_index(drop=True)

            df.index = df['date']
            t = df.groupby(self.id_col)['ret_1'].rolling(12).sum().reset_index().rename(columns={'ret_1': 'mom_f'})
            df = df.reset_index(drop=True).merge(t)

            ### Book to market
            df['year'] = df['date'].dt.year
            comp = self.load_compustat('quarterly').sort_values([self.id_col, 'date']).drop_duplicates().reset_index(drop=True)
            comp['year'] = comp['date'].dt.year
            comp['quarter'] = comp.index.values
            comp['quarter'] = comp.groupby([self.id_col, 'year'])['quarter'].transform('min')
            comp['quarter'] = comp.index.values - comp['quarter']
            comp['last_quarter'] = comp.groupby([self.id_col, 'year'])['quarter'].transform('max')
            comp = comp.loc[comp['quarter'] == comp['last_quarter'], ['year', self.id_col, 'asset', 'debt']].sort_values([self.id_col, 'year']).reset_index(drop=True)
            comp['asset_l'] = comp.groupby(self.id_col)['asset'].shift(1)

            # remove pure duplicates
            t = df[['permno', 'date']].duplicated()
            df = df.loc[~t, :]

            df = df.merge(comp, on=[self.id_col, 'year'], how='left')

            df['btm'] = 1000 * 1000 * (df['asset'] - df['debt']) / (df['shares_outstanding'] * df['S0'])

            ### investment
            df['inv'] = df['asset'] - df['asset_l']
            ### Profitability
            comp = self.load_compustat('quarterly').loc[:, ['date', self.id_col, 'asset', 'debt']].drop_duplicates().sort_values([self.id_col, 'date']).reset_index(drop=True)
            comp['book_equity'] = comp['asset'] - comp['debt']
            comp['book_equity'] = comp.groupby(self.id_col)['book_equity'].shift(1)
            # get the last quarter data only
            comp['year'] = comp['date'].dt.year
            comp['quarter'] = comp.index.values
            comp['quarter'] = comp.groupby([self.id_col, 'year'])['quarter'].transform('min')
            comp['quarter'] = comp.index.values - comp['quarter']
            comp['last_quarter'] = comp.groupby([self.id_col, 'year'])['quarter'].transform('max')
            comp = comp.loc[comp['quarter'] == comp['last_quarter'], ['year', self.id_col, 'book_equity']].sort_values([self.id_col, 'year']).reset_index(drop=True)
            df = df.merge(comp, on=[self.id_col, 'year'], how='left')

            df.index = df['date']
            t = df.groupby(self.id_col).apply(lambda x: x['book_equity'].fillna(method='ffill').fillna(method='bfill')).reset_index().drop_duplicates()
            df = df.reset_index(drop=True)
            del df['book_equity']
            df = df.merge(t, how='left')

            comp = self.load_compustat('yearly').drop_duplicates()
            comp['year'] = comp['date'].dt.year
            comp = comp.drop(columns='date')
            df = df.merge(comp)
            df['prof'] = df['income'] / df['book_equity']
            final = df[[self.id_col, 'date'] + var_list].copy()
            print(pd.isna(final).mean())

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
            final = final[[self.id_col, 'date'] + var_list].copy()
            final.to_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp_compustat.p')
        else:
            final = pd.read_pickle(self.par.data.dir + f'{self.name}/int/pred_crsp_compustat.p')
        return final

    def get_beta(self, freq='daily'):
        print('#' * 50)
        print('Start Beta', freq)
        print('#' * 50, flush=True)
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
        t = t.reset_index().melt(id_vars=['date']).rename(columns={'variable': self.id_col, 'value': f'beta_{freq}'})
        return t

    # def load_rf(self, reload=False):
    # if reload:
    #     df = pd.read_csv(self.par.data.dir + 'raw/rf.csv')
    #     df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    #     df['one'] = 1
    #     final = []
    #     for d in df['date'].unique():
    #         temp = df.loc[df['date'] == d, :]
    #
    #         m = KNeighborsRegressor(n_neighbors=2, weights='distance').fit(temp[['one', 'days']], temp['rate'])
    #
    #         rf = m.predict(pd.DataFrame(data={'one': [1], 'days': [28]}))[0] / 100
    #         final.append([d, rf])
    #
    #     df = pd.DataFrame(final)
    #     df.to_pickle(self.par.data.dir + 'raw_merge/rf.p')
    #
    #     df = pd.read_pickle(self.par.data.dir + 'raw_merge/rf.p')
    #     return df

    def martin_wagner_var_mkt(self, reload=False):
        if reload:
            if self.par.data.opt_smooth in [OptSmooth.VOLA_CUBIC]:

                df = pd.read_csv(self.par.data.dir + 'raw/spx_opt_surf.csv')
                df = df.rename(columns={'days': 'T', 'cp_flag': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares', 'impl_strike': 'strike',
                                        's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'impl_premium': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                        'expiration': 'exdate'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df['exdate'] = df['date'] + pd.DateOffset(days=df['T'].iloc[0])
                df['o_bid'] = df['o_ask']
                df['option_id'] = df.index
                df['strike'] *= 10000

                df[self.id_col] = df['ticker']
            else:
                df = pd.read_csv(self.par.data.dir + 'raw/spx_opt.csv')

                df = df.rename(columns={'t': 'T', 'cp_flag': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares', 'strike_price': 'strike',
                                        's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                        'expiration': 'exdate'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
                df['T'] = (df['exdate'] - df['date']).dt.days
                df[self.id_col] = df['ticker']

            price = pd.read_csv(self.par.data.dir + 'raw/spx_price.csv').rename(columns={'close': 'S'})[['date', 'S']]
            price['date'] = pd.to_datetime(price['date'], format='%Y%m%d')
            df = df.merge(price)

            ##################
            # clean option
            ##################

            df = df.rename(columns={'t': 'T', 'call_put': 'cp', 'total_open_interest': 'open_interest', 'shares_outstanding': 'shares',
                                    's_close': 'S', 'gv_key': 'gvkey', 'implied_volatility': 'impl_volatility', 'best_offer': 'o_ask', 'best_bid': 'o_bid', 'option_id': 'optionid',
                                    'expiration': 'exdate'})

            df['cp'] = df['cp'].astype('str')

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
            if ind.mean() != 0.0:
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
            # for i in range(10)
            for i in range(t.shape[0]):
                d = t.iloc[i, 0]
                g = t.iloc[i, 1]

                temp = df.loc[(df['date'] == d) & (df[self.id_col] == g), :].copy().reset_index(drop=True)

                def trapezoidal_integral_approx(t, y):
                    return np.sum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2.)

                S = temp['S'].iloc[0]
                v = 2 * trapezoidal_integral_approx(temp['strike'].values, temp['opt_price'].values) / S ** 2
                res.append([d, g, v])

                if i % 1000 == 0:
                    print(i, '/', t.shape[0])
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
            print(df.groupby('year').apply(r2), flush=True)

            df[['date', self.id_col, 'MW']].to_pickle(self.par.data.dir + f'{self.name}/int/MW.p')

        df = pd.read_pickle(self.par.data.dir + f'{self.name}/int/MW.p')
        return df

    def do_batch_hist_theta(self, b):
        print('Start batch', b, flush=True)
        self.historical_theta(reload=True, batch_id=b)
        print('Done', b)


    def find_batch_vilknoy_thetta(self, b):

        temp_dir = self.par.data.dir + 'temp_hist_theta/'
        save_dir = self.par.data.dir + 'temp_vilknoy_theta/'
        os.makedirs(save_dir,exist_ok=True)
        label = pd.read_pickle(temp_dir + f'batch_label_{b}.p')
        ind = label.sort_values([self.id_col, 'date']).index
        m = pd.read_pickle(temp_dir + f'batch_m_{b}.p')
        m = m.loc[ind, :].reset_index(drop=True)
        label = label.loc[ind, :].reset_index(drop=True)

        print('data loaded for batch ', b, flush=True)

        ##################
        # add vilknoy to label
        ##################
        them = pd.read_csv(f'{self.par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})

        them['date'] = pd.to_datetime(them['date'])
        them['permno'] = them['permno'].astype(int)
        them['glb2_D30'] = them['glb2_D30'] / 12
        them['glb3_D30'] = them['glb3_D30'] / 12


        label=label.merge(them[['date','permno','glb2_D30']],how='left')

        res = pd.DataFrame()
        for i in range(m.shape[0]):
        # for i in range(100):
            d = label.iloc[i, 0]
            g = label.iloc[i, 1]
            ind = (label['date'] == d) & (label[self.id_col] == g)

            temp = m.loc[ind, :].copy()
            temp['theta'] = 0.0
            temp = temp.values
            y = label.loc[ind, 'glb2_D30'].copy().iloc[0]

            def func(theta):
                temp[:, -1] = theta
                r = np.apply_along_axis(Econ.g_apply_log, axis=1, arr=temp)[0]
                return np.mean(np.square(r*1000 - y*1000))

            b_nd = opti.Bounds([0.0], [1.0])
            try:
                r = opti.minimize(fun=func, x0=0.5, method='trust-constr', bounds=b_nd)
                theta = r['x'][0]
                func(theta)
                m_pred = m.iloc[i, :]
                m_pred['theta'] = theta
            except:
                theta = np.nan
            r = pd.Series({'date': d, self.id_col: g, 'theta_v': theta})
            print(i,'/',label.shape[0],flush=True)
            res=res.append(r,ignore_index=True)

        res.to_pickle(save_dir+f'vilkn_{b}.p')




    def historical_theta(self, reload=False, create_batch=False, batch_id=0):
        NB_BATCH = 25
        temp_dir = self.par.data.dir + 'temp_hist_theta/'
        if create_batch:
            self.load_final()

            d = np.array(self.label_df['permno'].unique())
            bs = int(np.floor(d.shape[0] / NB_BATCH))
            for b in range(NB_BATCH + 1):
                start = b * bs
                end = min((b + 1) * bs, d.shape[0])
                ind = self.label_df['permno'].isin(d[start:end])

                self.m_df.loc[ind, :].to_pickle(temp_dir + f'batch_m_{b}.p')
                self.label_df.loc[ind, :].to_pickle(temp_dir + f'batch_label_{b}.p')
                print('Saved batch', b, flush=True)
            print('All batch created', flush=True)

        if reload:
            os.makedirs(temp_dir, exist_ok=True)

            label = pd.read_pickle(temp_dir + f'batch_label_{batch_id}.p')
            ind = label.sort_values([self.id_col, 'date']).index
            m = pd.read_pickle(temp_dir + f'batch_m_{batch_id}.p')
            m = m.loc[ind, :].reset_index(drop=True)
            label = label.loc[ind, :].reset_index(drop=True)
            res = []
            prb_i = []
            i = 1000
            print('data loaded for batch ', batch_id, flush=True)

            def compute_theta_for_index_i(i):
                d = label.iloc[i, 0]
                g = label.iloc[i, 1]
                ind = (label['date'] < d) & (label['date'] >= (d - pd.DateOffset(years=1))) & (label[self.id_col] == g)
                if (ind.sum() > 26):  # ask for a minimum of 6 month data
                    temp = m.loc[ind, :].copy()
                    temp['theta'] = 0.0
                    temp = temp.values
                    y = label.loc[ind, 'log_ret'].copy()
                    y = np.mean(y)

                    def func(theta):
                        temp[:, -1] = theta
                        t = np.apply_along_axis(Econ.up_down_apply_log, axis=1, arr=temp)
                        S = temp[:, -2]
                        t[:, 0] = t[:, 0] / S ** theta
                        t[:, 1] = t[:, 1] / S ** theta
                        t = np.mean(t, 0)
                        r = t[0] / t[1]
                        r = r - np.mean(np.log(S))
                        return np.mean(np.square(r - y))

                    b = opti.Bounds([-5.0], [5.0])
                    try:
                        r = opti.minimize(fun=func, x0=0.5, method='trust-constr', bounds=b)
                        theta = r['x'][0]
                        m_pred = m.iloc[i, :]
                        m_pred['theta'] = theta
                        pred = Econ.g_apply_log(m_pred.values).numpy()
                        m_pred['theta'] = 1.0
                        pred_0 = Econ.g_apply_log(m_pred.values).numpy()
                    except:
                        pred = np.nan
                        theta = np.nan
                        pred_0 = np.nan
                        print(f'Opti problem with {i, d, g}', flush=True)
                        prb_i.append(i)
                else:
                    pred = np.nan
                    theta = np.nan
                    pred_0 = np.nan

                r = pd.Series({'date': d, self.id_col: g, 'theta': theta, 'pred': pred, 'pred_0': pred_0})
                return r

            save_df = pd.DataFrame()
            for i in range(m.shape[0]):
                t = compute_theta_for_index_i(i)
                if t is not None:
                    save_df = save_df.append(t, ignore_index=True)
                if i % 1 == 0:
                    print(i, '/', m.shape[0], flush=True)

            save_df.to_pickle(temp_dir + f'temp_{batch_id}.p')
            print('save batch', batch_id, flush=True)

        df = pd.read_pickle(self.par.data.dir + 'theta_hist.p')
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
        ind = (df['impl_volatility'] > 0) & (df['delta'].abs() < 0.5)
        df = df.loc[ind, :]

        rf = self.load_rf()
        rf.columns = ['date', 'rf']
        df = df.merge(rf)

        def BlackScholes_price(sigma, S, r, K):
            dt = 28 / 365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S > K] = pr_put[S > K]
            return pr

        def BlackScholes_error(sigma, S, r, K, mid):
            dt = 28 / 365
            Phi = stats.norm(loc=0, scale=1).cdf

            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
            d2 = d1 - sigma * np.sqrt(dt)

            pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
            pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
            pr[S > K] = pr_put[S > K]
            return np.square(pr - mid)

        def find(d):
            x_init = d['impl_volatility']
            S = d['S0']
            r = d['rf']
            K = d['strike']
            pr = d['opt_price']
            try:
                res = opti.minimize(BlackScholes_error, x_init, args=(S, r, K, pr))
                iv = res.x
            except:
                iv = np.nan
            return iv[0]

        # pandarallel.initialize(progress_bar=True)
        # r= df.parallel_apply(find,axis=1)
        # r = pd.read_pickle('temp.p')
        # r = r.values
        # r.shape
        # df['impl_volatility'] =r
        # df.to_pickle(self.par.data.dir + f'{self.name}/int/opt.p')
        #
        # p=BlackScholes_price(df['impl_volatility'],df['S0'],df['rf'],df['strike'])
        # print('Error', (p-df['opt_price']).abs().mean())

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

        if self.par.data.opt_smooth in [OptSmooth.VOLA_CUBIC]:
            self.clean_vola_surface()
        else:
            self.clean_opt_all()

        self.martin_wagner_var_mkt(reload=True)
        self.marting_wagner_return(reload=True)
        print('start', flush=True)

        if self.par.data.crsp:
            if self.par.data.comp:
                self.load_pred_compustat_and_crsp(reload=True)
            else:
                self.load_pred_crsp_only(reload=True)

        # self.historical_theta(reload=True)
        self.create_a_dataset()
