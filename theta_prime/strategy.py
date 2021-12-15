import pandas as pd
import numpy as np
from data import Data
import statsmodels.api as sm
from matplotlib import pyplot as plt
import os
import didipack as didi
import shutil
from parameters import *
import pandas as pd
import numpy as np
from parameters import *
import didipack as didi
from cleaner import Cleaner
import urllib.request
import zipfile
import os
import logging

logging.basicConfig(level=logging.DEBUG)


class Strategy:
    def __init__(self, strat_name='my strategy',author = '', par=Params()):
        self.strat_name = strat_name
        self.author = author
        self.par = par
        self.data = Data(par)
        self.ff_dir = 'data/ff/'

    def _download_ff(self):
        os.makedirs(self.ff_dir, exist_ok=True)
        ff_five_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        ff_mom_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'

        def load_and_extract(ff_url):
            urllib.request.urlretrieve(ff_url, self.ff_dir + 'ff5.zip')
            zip_file = zipfile.ZipFile(self.ff_dir + 'ff5.zip', 'r')
            zip_file.extractall(path=self.ff_dir)
            zip_file.close()

        load_and_extract(ff_five_url)
        load_and_extract(ff_mom_url)

    def load_ff(self):
        if os.path.exists(self.ff_dir + 'F-F_Research_Data_5_Factors_2x3_daily.CSV'):
            ff = self._load_ff()
        else:
            self._download_ff()
            ff = self._load_ff()
        ff['mkt'] = ff['mkt-rf'] + ff['rf']

        return ff

    def _load_ff(self):
        ff_5 = pd.read_csv(self.ff_dir + 'F-F_Research_Data_5_Factors_2x3_daily.CSV', skiprows=3)
        ff_mom = pd.read_csv(self.ff_dir + 'F-F_Momentum_Factor_daily.CSV', skiprows=12)
        ff_mom = ff_mom.iloc[:-1, :]
        c = [x.lower().replace(' ', '') for x in ff_5.columns]
        c[0] = 'date'
        ff_5.columns = c
        c = [x.lower().replace(' ', '') for x in ff_mom.columns]
        c[0] = 'date'
        ff_mom.columns = c
        ff_mom['date'] = pd.to_datetime(ff_mom['date'], format='%Y%m%d')
        ff_5['date'] = pd.to_datetime(ff_5['date'], format='%Y%m%d')
        ff = ff_mom.merge(ff_5)

        for c in ff.columns[1:]:
            ff[c] /= 100
        return ff

    def portfolio_quantile_sort(self, df,Q, rebalance_freq=1,ret_col='R',signal = 'signal',w_col=None):
        assert type(df) == pd.DataFrame, "df needs to be a pandas dataframe"
        assert signal in df.columns, "df needs to be a pandas dataframe with a columns named 'signal'"
        assert ret_col in df.columns, f"df needs to be a pandas dataframe with a columns named {ret_col} with the returns"
        D = df[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
        D['reb'] = False
        D.loc[0, ['reb']] = True
        D.loc[::rebalance_freq, 'reb'] = True
        df = df.merge(D)

        df = df.sort_values(['permno', 'date']).reset_index(drop=True)
        df['long'] = np.nan
        df.loc[df['reb'], 'long'] = df.loc[df['reb'], :].groupby('date')[signal].apply(lambda x: pd.qcut(x, Q, labels=False, duplicates='drop'))
        if pd.isna(df['long']).sum()>0:
            df['long'] = df.groupby('permno')['long'].fillna(method='ffill')

        if w_col is None:
            df=df.groupby(['long','date'])[ret_col].mean().reset_index().pivot(columns='long',index='date',values=ret_col)
        else:
            df['w*r'] = df[w_col]*df[ret_col]
            df = df.groupby(['long', 'date'])[['w*r',w_col]].sum().reset_index()
            df[ret_col] = df['w*r']/df[w_col]
            df=df.pivot(columns='long',index='date',values=ret_col)
        df.columns.name = 'Q'
        return df




    def portfolio_from_signal(self, df,
                              signal ='signal',
                              ret_col ='R',
                              rebalance_freq=1,
                              w_col=None,
                              nb_stock_long=-1,
                              nb_stock_short=-1,
                              Q_long=-1,
                              Q_short=-1,
                              hard_min=10
                              ):

        assert type(df) == pd.DataFrame, "df needs to be a pandas dataframe"
        assert signal in df.columns, "df needs to be a pandas dataframe with a columns named 'signal'"
        assert ret_col in df.columns, f"df needs to be a pandas dataframe with a columns named {ret_col} with the returns"
        assert not ((nb_stock_long > 0) & (Q_long > 0)), "Please select either a number of stocks long nb_stock_long or long quantile to buy Q_long, not both"
        assert not ((nb_stock_short > 0) & (Q_short > 0)), 'Please select either a number of stocks short nb_stock_short or short quantile to buy Q_short, not both'
        assert not ((nb_stock_long < 0) & (Q_long < 0) & (nb_stock_short < 0) & (Q_short < 0)), 'Please select either a number of stocks nb_stock or quantile to buy Q (either long or short, or both)'

        # we start on the first date on the dataframe, and rebalance every D days
        D = df[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
        D['reb'] = False
        D.loc[0, ['reb']] = True
        D.loc[::rebalance_freq, 'reb'] = True
        df = df.merge(D)

        df = df.sort_values(['permno', 'date']).reset_index(drop=True)

        if Q_long > -1:
            df['long'] = np.nan
            df.loc[df['reb'], 'long'] = df.loc[df['reb'], :].groupby('date')[signal].apply(lambda x: pd.qcut(x, Q_long, labels=False, duplicates='drop'))
            if pd.isna(df['long']).sum() > 0:
                df['long'] = df.groupby('permno')['long'].fillna(method='ffill')
            df['long'] = (df['long'] == Q_long - 1) * 1

        if Q_short > -1:
            df['short'] = np.nan
            df.loc[df['reb'], 'short'] = df.loc[df['reb'], :].groupby('date')[signal].apply(lambda x: pd.qcut(x, Q_short, labels=False, duplicates='drop'))
            if pd.isna(df['short']).sum() > 0:
                df['short'] = df.groupby('permno')['short'].fillna(method='ffill')
            df['short'] = (df['short'] == 0.0) * 1

        if nb_stock_long > -1:
            long = df.loc[df['reb'], ['date', 'permno', signal]].sort_values(['date', signal], ascending=False).groupby('date').head(nb_stock_long)
            long['long'] = 1
            del long[signal]
            df = df.merge(long, how='left')
            df.loc[(df['reb'] & (df['long'] != 1)), 'long'] = 0.0
            if pd.isna(df['long']).mean() > 0:
                for i in range(rebalance_freq - 1):
                    df['t'] = df.groupby('permno')['long'].shift(1)
                    ind = (~df['reb']) & (pd.isna(df['long']))
                    df.loc[ind, 'long'] = df.loc[ind, 't']
                del df['t']
            del long

        if nb_stock_short > -1:
            short = df.loc[df['reb'], ['date', 'permno', signal]].sort_values(['date', signal], ascending=False).groupby('date').tail(nb_stock_short)
            short['short'] = 1
            del short[signal]
            df = df.merge(short, how='left')
            df.loc[(df['reb'] & (df['short'] != 1)), 'short'] = 0.0
            if pd.isna(df['short']).mean() > 0:
                for i in range(rebalance_freq - 1):
                    df['t'] = df.groupby('permno')['short'].shift(1)
                    ind = (~df['reb']) & (pd.isna(df['short']))
                    df.loc[ind, 'short'] = df.loc[ind, 't']
                del df['t']
            del short

        if w_col is None:
            # computing returns with equally weighted portfolio
            if 'long' in df.columns:
                ind = df['long'] == 1
                long = df.loc[ind, :].groupby('date')[[ret_col, 'long']].sum()
                long['ret'] = long[ret_col] / long['long']
                # set to 0 whenever we have less stock in protoflio than hard_min --> we skip that day
                long.loc[long['long'] <= hard_min, 'ret'] = 0
                long = long[['long', 'ret']]
                long.columns = ['nb_long', 'ret_long']
            if 'short' in df.columns:
                ind = df['short'] == 1
                short = df.loc[ind, :].groupby('date')[[ret_col, 'short']].sum()
                short['ret'] = short[ret_col] / short['short']
                # set to 0 whenever we have less stock in protoflio than hard_min --> we skip that day
                short.loc[short['short'] <= hard_min, 'ret'] = 0
                short = short[['short', 'ret']]
                short['ret'] *= -1
                short.columns = ['nb_short', 'ret_short']
        else:
            # computing returns with protfolio weighted by w_col
            df['w*R'] = df[ret_col] * df[w_col]
            if 'long' in df.columns:
                ind = df['long'] == 1
                long = df.loc[ind, :].groupby('date')[['w*R', 'long', w_col]].sum()
                long['ret'] = long['w*R'] / long[w_col]
                # set to 0 whenever we have less stock in protoflio than hard_min --> we skip that day
                long.loc[long['long'] <= hard_min, 'ret'] = 0
                long = long[['long', 'ret']]
                long.columns = ['nb_long', 'ret_short']
            if 'short' in df.columns:
                ind = df['short'] == 1
                short = df.loc[ind, :].groupby('date')[['w*R', 'short', w_col]].sum()
                short['ret'] = short['w*R'] / short[w_col]
                # set to 0 whenever we have less stock in protoflio than hard_min --> we skip that day
                short.loc[short['short'] <= hard_min, 'ret'] = 0
                short = short[['short', 'ret']]
                short['ret'] *= -1
                short.columns = ['nb_short', 'ret_short']

        if ('long' in df.columns) & ('short' not in df.columns):
            ret = long.reset_index()
            ret.columns = ['date', 'nb_stock', 'ret']
        if ('short' in df.columns) & ('long' not in df.columns):
            ret = short.reset_index()
            ret.columns = ['date', 'nb_stock', 'ret']

        if ('short' in df.columns) & ('long' in df.columns):
            long.columns = ['nb_long', 'ret_long']
            short.columns = ['nb_short', 'ret_short']
            ret = long.reset_index().merge(short.reset_index())
            ret['ret'] = ret['ret_long'] + ret['ret_short']

        return ret

    def merge_ret_with_ff(self, ret):
        """
        add the fama french returns corresponding to the date gap.
        We take the daily ff factors and create the cumulated returns between the appropriate dates.
        :param ret:
        :return:
        """
        ff = self.load_ff()
        D = ret[['date']].drop_duplicates().reset_index(drop=True)
        D['t'] = D.index
        ff = ff.loc[(ff['date'] <= D['date'].max()) & (ff['date'] >= D['date'].min()), :]
        ff = ff.merge(D, how='left')
        ff['t'] = ff['t'].fillna(method='ffill')
        C = ff.columns[1:-1]
        for c in C:
            ff[c] += 1
        ff = ff.groupby('t')[C].prod()
        for c in C:
            D[c] = ff[c] - 1
        del D['t']

        df = ret.merge(D)
        df['alpha'] = 1.0
        return df


    def fama_reg(self, ret, verbose=False):
        df = self.merge_ret_with_ff(ret)
        m1 = sm.OLS(df['ret'] - df['rf'], df[['alpha', 'mkt-rf']]).fit()
        m3 = sm.OLS(df['ret'] - df['rf'], df[['alpha', 'mkt-rf', 'hml', 'smb']]).fit()
        m5 = sm.OLS(df['ret'] - df['rf'], df[['alpha', 'mkt-rf', 'hml', 'smb', 'rmw', 'cma']]).fit()
        m6 = sm.OLS(df['ret'] - df['rf'], df[['alpha', 'mkt-rf', 'hml', 'smb', 'rmw', 'cma', 'mom']]).fit()
        M = [m1, m3, m5, m6]
        if verbose:
            print(m6.summary2())
        return M

    def _annualize_factor(self, ret):
        return np.round(365 / ret['date'].diff().dt.days.mean(), 1)

    def cumprod(self, ret, fig_dir=None):
        df = self.merge_ret_with_ff(ret)
        df.index = df['date']
        df = (1 + df[['ret', 'mkt']]).cumprod()

        if fig_dir is not None:
            k = 0
            df['ret'].plot(label='Strategy', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            k = +1
            df['mkt'].plot(label='Market', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.xlabel('Date')
            plt.ylabel('Return')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir)
            plt.close()
        return df

    def rolling_sharp(self, ret, fig_dir):
        fact = int(np.round(self._annualize_factor(ret)))
        df = self.merge_ret_with_ff(ret)
        df.index = df['date']

        def get_sharp(x):
            m = (1 + x).prod() ** (fact / x.shape[0]) - 1
            std = x.std() * np.sqrt(fact)
            return m / std

        df = df[['ret', 'mkt']].rolling(fact * 5).apply(get_sharp)

        if fig_dir is not None:
            k = 0
            df['ret'].plot(label='Strategy', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            k = +1
            df['mkt'].plot(label='Market', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.xlabel('Date')
            plt.ylabel('5y-rolling SR')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir)
            plt.close()

        return df

    def rolling_mean(self, ret, fig_dir):
        fact = int(np.round(self._annualize_factor(ret)))
        df = self.merge_ret_with_ff(ret)
        df.index = df['date']

        def get_mean(x):
            m = (1 + x).prod() ** (fact / x.shape[0]) - 1
            return m

        df = df[['ret', 'mkt']].rolling(fact * 5).apply(get_mean)

        if fig_dir is not None:
            k = 0
            df['ret'].plot(label='Strategy', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            k = +1
            df['mkt'].plot(label='Market', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.xlabel('Date')
            plt.ylabel('5y-rolling mean return')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir)
            plt.close()

        return df

    def table_quantile_mean_std_sharp_pred(self, t,t_pred =None, annualize = -1,in_percentage = True):
        if t_pred is None:
            t_pred =t.copy()
            skip_pred = True
        else:
            skip_pred =False

        if t.columns[0] == 0:
            t.columns += 1
            t_pred.columns += 1
        t = t.rename(columns={t.columns.max(): 'High(H)', t.columns.min(): 'Low(L)'})
        t_pred = t_pred.rename(columns={t_pred.columns.max(): 'High(H)', t_pred.columns.min(): 'Low(L)'})

        t['H-L'] = t.iloc[:, -1] - t.iloc[:, 0]
        t_pred['H-L'] = t_pred.iloc[:, -1] - t_pred.iloc[:, 0]

        p = t_pred.mean()
        m = t.mean()
        s = t.std()
        if annualize > 0:
            m = ((1 + m) ** annualize) - 1
            p = ((1 + p) ** annualize) - 1
            s *= np.sqrt(annualize)
        sharp = m / s

        if in_percentage:
            p *= 100
            m *= 100
            s *= 100

        if skip_pred:
            res = pd.DataFrame({'Avg': m, 'SD': s, 'SR': sharp}).round(2)
        else:
            res = pd.DataFrame({'Pred': p, 'Avg': m, 'SD': s, 'SR': sharp}).round(2)
        return res

    def rolling_std(self, ret, fig_dir):
        fact = int(np.round(self._annualize_factor(ret)))
        df = self.merge_ret_with_ff(ret)
        df.index = df['date']

        def get_std(x):
            std = x.std() * np.sqrt(fact)
            return std

        df = df[['ret', 'mkt']].rolling(fact * 5).apply(get_std)

        if fig_dir is not None:
            k = 0
            df['ret'].plot(label='Strategy', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            k = +1
            df['mkt'].plot(label='Market', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.xlabel('Date')
            plt.ylabel('5y-rolling STD')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir)
            plt.close()

        return df

    def finance_stat(self, ret, verbose=False):
        fact = self._annualize_factor(ret)

        def get_finance_stat(x):
            m = (1 + x).prod() ** (fact / x.shape[0]) - 1
            std = x.std() * np.sqrt(fact)
            return pd.Series({'mean': m, 'std': std, 'sharp': m / std})

        ret = self.merge_ret_with_ff(ret)
        res = ret[['ret', 'mkt']].apply(get_finance_stat)
        if verbose:
            print(res.round(3))
        return res


    def _create_paper(self,save_dir,pdf_name='default_pdf_name'):
        paper = didi.LatexPaper(dir_= save_dir)
        # Creating the paper itself with a given title, author name and some default sections (non-mandatory)
        paper.create_paper(pdf_name, title=f"Strategy Report: {self.strat_name}", author=self.author, sec=['Performance'])
        return paper


    def academic_report(self, ret, intro='',save_dir='report/academic/',pdf_name='academic'):
        paper = self._create_paper(save_dir=save_dir,pdf_name=pdf_name)

        df = ret.copy()
        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month
        M = self.fama_reg(ret)
        table = didi.TableReg()
        for m in M:
            table.add_reg(m)
        table.save_tex(paper.dir_tables + 'ff_reg.tex')

        res = self.finance_stat(ret)
        res.columns = [x.replace('_', ' ') for x in res.columns]
        res.round(3).to_latex(paper.dir_tables + 'perf_summary.tex', escape=False)

        res = self.finance_stat(ret.loc[res['date'].dt.year >= 2010, :])
        res.columns = [x.replace('_', ' ') for x in res.columns]
        res.round(3).to_latex(paper.dir_tables + 'perf_summary_recent.tex', escape=False)

        _ = self.cumprod(ret, paper.dir_figs + 'overall_perf_cum.png')
        _ = self.cumprod(ret.loc[ret['date'].dt.year >= 2010, :], paper.dir_figs + 'post2010_perf_cum.png')

        _ = self.rolling_sharp(ret, paper.dir_figs + 'rolling_sharp_5y.png')
        _ = self.rolling_mean(ret, paper.dir_figs + 'rolling_mean_5y.png')
        _ = self.rolling_std(ret, paper.dir_figs + 'rolling_std_5y.png')

        if intro is not None:
            paper.append_text_to_sec('Performance', intro)

        paper.append_table_to_sec(table_name='perf_summary', sec_name='Performance', caption='The table below shows the mean, standard deviation and sharp ratio of the equally, and market-cap. weighted strategy, as well as the market. '
                                                                                                  'All numbers are estimated on the full sample.')
        paper.append_table_to_sec(table_name='perf_summary_recent', sec_name='Performance', caption=r'The table below shows same stats as table \ref{table:perf_summary} but on the subsample starting in 2010.')

        paper.append_fig_to_sec(fig_names=['overall_perf_cum', 'post2010_perf_cum'], sec_name='Performance',
                                     main_caption=r"The figures above show the cummulative return of the equally, and market-cap. weighted strategy, as well as the market. "
                                                  r"Panel (a) uses the full sample while panel (b) assumes trading start on 2010.")

        paper.append_fig_to_sec(fig_names=['rolling_std_5y', 'rolling_mean_5y', 'rolling_sharp_5y'], sec_name='Performance',
                                     main_caption=r"The figures above show the mean, std, and sharp ratio of the equally weighted, and market-cap. weighted strategy. All figure "
                                                  r"uses a rolling window of 5 year. ")

        paper.append_table_to_sec(table_name='ff_reg', sec_name='Performance', caption='The table below shows the Fama French 3 factor regression for the strategy weighted with market capitalisation (1), and equally weighted (2). ')

        paper.append_fig_to_sec(fig_names=['number_stocks_long'], sec_name='Performance',
                                     main_caption=r"The figures above show the number of stocks in the portfolio throughout the sample.")

        # inputted_path = r"\input{" + self.paper.dir + self.pdf_name+"}"
        # inputted_path = self.paper.dir +
        old_dir = os.getcwd()
        dir = old_dir + '/' + paper.dir
        tex = pdf_name + ".tex"
        os.chdir(dir)
        os.system("pdflatex %s" % tex)
        os.system(f'scp {tex.replace(".tex", ".pdf")} ../PDF/')
        os.chdir(old_dir)


self = Strategy(Params())
