import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import didipack as didi
import os
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import shutil
import tensorflow as tf
import matplotlib
from strategy import Strategy

par = Params()

class Trainer:
    def __init__(self, par=Params()):
        self.par = par
        # self.par.name = 'defaultL64_32_16_Lr001Dropout001BS512ActreluOutRange05LossMSERetLOGd3OptCompCrspEXT'
        self.model = NetworkMean(self.par)

        if self.par.data.H == 20:
            name_ret = 'ret1m'
        if self.par.data.H == 60:
            name_ret = 'ret3m'
        if self.par.data.H == 120:
            name_ret = 'ret6m'
        self.name_ret = name_ret

    def launch_training_expanding_window(self):
        np.random.seed(12345)
        tf.random.set_seed(12345)
        self.model.data.load_internally()
        print('DATA example')
        print(self.model.data.x_df.head())
        print(self.model.data.x_df.describe())

        YEAR = range(1996, 2020)

        L = [int(x.split('perf_')[1].split('.p')[0]) for x in os.listdir(self.model.res_dir) if 'perf_' in x]
        Y = []
        for y in YEAR:
            if y not in L:
                Y.append(y)
            else:
                print('already run', y)
        YEAR = Y
        print('Run on years', YEAR)

        for year in YEAR:
            try:
                self.model.run_year(year)
            except:
                print('skip year', year, flush=True)

    def create_paper(self):
        # define the paper maine directory (can be relative or abolute path)
        f_dir = self.par.model.tex_dir + '/' + str(self.par.name) + '/'
        if os.path.exists(f_dir):
            shutil.rmtree(f_dir)
        paper = didi.LatexPaper(dir_=f_dir)
        # Creating the paper itself with a given title, author name and some default sections (non-mandatory)
        paper.create_paper(self.par.model.tex_name, title="KiSS: Keep it Simple Stupid", author="Antoine Didisheim", sec=['Introduction', 'Results'])

        L = "Layer: ("
        for l in self.par.model.layers:
            L += str(l) + ','
        L = L[:-1] + ')'
        model_description = '\n' + "In this section we present the results where the neural network architecture is defined as, \n \n" \
                                   r"\begin{enumerate}" + '\n' \
                                                          rf"\item Learning rate {self.par.model.learning_rate}" + '\n' \
                                                                                                                   rf"\item Optimizer {self.par.model.opti}" + '\n' \
                                                                                                                                                               rf"\item {L}" + '\n' \
                                                                                                                                                                               rf"\item Loss {self.par.model.loss.name}" + '\n' \
                                                                                                                                                                                                                           rf"\item Output range {self.par.model.output_range}" + '\n' \
                                                                                                                                                                                                                                                                                  rf"\item Output positive only {self.par.model.output_pos_only}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                    rf"\item Batch size {self.par.model.batch_size}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                       rf"\item Training data {self.par.data.cs_sample.name}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                rf"\item Forecasting horizon {self.par.data.H}" + '\n' + \
                            r"\end{enumerate}" + '\n'
        paper.append_text_to_sec(sec_name='Introduction', text=model_description.replace('_', ' '))

        # MODEL_LIST = ['pred', 'vilk', 'mw']
        MODEL_LIST = ['pred', 'mw']
        MODEL_ALL = MODEL_LIST + ['NN4EW']
        m_dict = {'pred': 'DNN', 'vilk': 'vilk', 'mw': 'Martin&Wagner', 'NN4EW': 'Gu&Kelly'}

        L = [x for x in os.listdir(self.model.res_dir) if 'perf_' in x]
        print('look for data in dir', self.model.res_dir)
        print('found', L)
        full_df = pd.DataFrame()
        for l in tqdm(L, 'load original df'):
            full_df = full_df.append(pd.read_pickle(self.model.res_dir + l))


        ## add martin wagner
        mw = self.model.data.load_mw()
        full_df = full_df.merge(mw, how='left')
        true_full = full_df.copy()

        full_df = full_df.loc[~pd.isna(full_df['mw']), :]
        # full_df = full_df.loc[~pd.isna(full_df['vilk']), :]
        full_df = full_df.reset_index(drop=True)

        name_ret = self.name_ret

        def r2(df_, col='pred'):
            if self.par.data.H == 20:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - 0.0) ** 2).sum()
            else:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - (1.06 ** (self.par.data.H / 252) - 1)) ** 2).sum()
            return r2_pred

        def r2_against_cs(df_, col='pred'):
            if self.par.data.H == 20:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - df_.groupby('date')[col].transform('mean')) ** 2).sum()
            return r2_pred

        def r2_against_yearly(df_, col='pred'):
            if self.par.data.H == 20:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - df_.groupby('year')[col].transform('mean')) ** 2).sum()
            return r2_pred

        def r2_of_cs(df_, col='pred'):
            if self.par.data.H == 20:
                r2_pred = 1 - ((df_[name_ret] - df_.groupby('date')[col].transform('mean')) ** 2).sum() / ((df_[name_ret] - 0.0) ** 2).sum()
            return r2_pred

        # df[name_ret].mean()
        if self.par.data.H == 20:
            ADD_TEXT = r' $R^2$ is defined with 0.0 as a denominator benchmark. '
        else:
            ADD_TEXT = r' $R^2$ is defined with in sample mean as a denominator benchmark. '



        ##################
        # r2 comparing to kelly (FULL CROSS SECTION)
        ##################
        KELLY_MODEL_LIST = ['pred', 'NN4EW']
        k = self.model.data.load_kelly_bench()
        df = true_full.merge(k)
        temp = self.model.data.load_additional_crsp()
        df_strat = df.merge(temp)

        # temp = self.model.data.load_additional_crsp(reload=True)
        # df.merge(temp).to_pickle('res/res_kelly_2.p')
        # true_full.merge(temp).to_pickle('res/res_vilk_2.p')

        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in KELLY_MODEL_LIST:
                r[c] = r2(df.loc[ind, :], c)
            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(KELLY_MODEL_LIST):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'KELLY_cumulative_r2.png')
        self.plt_show()

        ##################
        # kelly benchmark smaller subsample
        ##################

        df = true_full.loc[~pd.isna(true_full['mw']), :].merge(k, how='outer')
        df = df.loc[~pd.isna(df['ret1m']), :]
        # df = df.loc[~pd.isna(df['mw']), :]

        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in MODEL_ALL:
                ind_na = ~pd.isna(df[c])
                r[c] = r2(df.loc[ind & ind_na, :], c)
            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(MODEL_ALL):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'KELLY_SMALL_cumulative_r2.png')
        self.plt_show()

        # We now add two figure side by side
        paper.append_fig_to_sec(fig_names=['KELLY_cumulative_r2', 'KELLY_SMALL_cumulative_r2'], sec_name='Results',
                                fig_captions=['Cumulative', 'Year per year'],  # you can add individual caption to each figure (leave empty for no label)
                                main_caption=r"The figures above show the $R^2$ of our model against some of the best model from Kelly's paper. "
                                             r"The first figure show the result on the large cross section of assets, the second only those present also in the vilknoy and MW papers. " + ADD_TEXT)

        df_kelly = df.copy()

        ##################
        # predict against the cross section prediction
        ##################

        df = true_full.loc[~pd.isna(true_full['mw']), :].merge(k, how='outer')
        df = df.loc[~pd.isna(df['ret1m']), :]
        # df = df.loc[~pd.isna(df['mw']), :]

        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in MODEL_ALL:
                ind_na = ~pd.isna(df[c])
                r[c] = r2_against_cs(df.loc[ind & ind_na, :], c)
            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(MODEL_ALL):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'small_vs_mean_pred.png')
        self.plt_show()

        ### vs pred by year
        df['year'] = df['date'].dt.year
        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in MODEL_ALL:
                ind_na = ~pd.isna(df[c])
                r[c] = r2_against_yearly(df.loc[ind & ind_na, :], c)
            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(MODEL_ALL):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'small_vs_mean_pred_year.png')
        self.plt_show()

        ### pred by month group
        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in MODEL_ALL:
                ind_na = ~pd.isna(df[c])
                r[c] = r2_of_cs(df.loc[ind & ind_na, :], c)
            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(MODEL_ALL):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'small_mean_pred.png')
        self.plt_show()

        # We now add two figure side by side
        paper.append_fig_to_sec(fig_names=['small_vs_mean_pred', 'small_vs_mean_pred_year'], sec_name='Results',
                                fig_captions=['vs month', 'vs year'],  # you can add individual caption to each figure (leave empty for no label)
                                main_caption=r"The figures above show the $R^2$ of the models measured against the mean prediction of the model. Panel a): daily average , and Panel b): yearly average.")

        df_kelly = df.copy()

        ##################
        # r2 plots
        ##################
        df = full_df.dropna().copy()
        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR, 'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            for c in MODEL_LIST:
                r[c] = r2(df.loc[ind, :], c)

            R.append(r)
        res = pd.DataFrame(R)
        res.index = res['year']
        del res['year']

        # cummulative r2 plots
        d = pd.to_datetime(res.index, format='%Y')
        for i, c in enumerate(MODEL_LIST):
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'$R^2_{oos}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'cumulative_r2.png')
        self.plt_show()

        # year per year
        df['year'] = df['date'].dt.year
        for i, c in enumerate(MODEL_LIST):
            t = df.groupby('year').apply(lambda x: r2(x, col=c))
            plt.plot(d, t, color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
        plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'Year per Year $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'year_r2.png')
        self.plt_show()

        # We now add two figure side by side
        paper.append_fig_to_sec(fig_names=['cumulative_r2', 'year_r2'], sec_name='Results',
                                fig_captions=['Cumulative', 'Year per year'],  # you can add individual caption to each figure (leave empty for no label)
                                main_caption=r"The figures above show the $R^2$ of the main models. " + ADD_TEXT)

        df_kelly['year'] = df_kelly['date'].dt.year
        tt = df_kelly.groupby('year')[MODEL_ALL].std()
        for k, c in enumerate(MODEL_ALL):
            print(k, c)
            plt.plot(tt.index, tt[c], color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k], label=m_dict[c])
            plt.grid()
        # plt.xlabel('Year')
        plt.ylabel(r'Pred. std')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'const_pred.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['const_pred'], sec_name='Results',

                                main_caption=r"The figure above show the standard deviation year per year of the predicitons. It is here to check constant predicitons. ")

        ##################
        # stock by stock
        ##################
        S = []
        for i, c in enumerate(MODEL_ALL):
            t = df_kelly.loc[~pd.isna(df_kelly[c]), :].groupby('permno').apply(lambda x: r2(x, col=c))
            plt.hist(t, color=didi.DidiPlot.COLOR[i], alpha=0.5, label=m_dict[c], bins=50, density=True)
            t.name = c
            S.append(t)
        plt.grid()
        plt.ylabel(r'$R^2 per Firm$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'hist_stock_r2.png')
        self.plt_show()

        # boxplot
        t = pd.DataFrame(S).T
        t.boxplot()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'box_stock_r2.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['hist_stock_r2', 'box_stock_r2'], sec_name='Results',
                                main_caption=r"The figures compare the $R^2$ firm by firm of each model with histograms (a) and boxplots (b)")

        ##################
        # correlation of perf
        ##################

        def corrfunc(x, y, ax=None, **kws):
            """Plot the correlation coefficient in the top left hand corner of a plot."""
            r, _ = pearsonr(x, y)
            ax = ax or plt.gca()
            ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

        t = t.fillna(0.0)
        t.columns = [m_dict[c] for c in t.columns]
        g = sns.pairplot(t)
        g.map_lower(corrfunc)
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'corr_firms.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['corr_firms'], sec_name='Results', size="80mm",
                                main_caption=r"The figures illustrate the correlation between each models firm by firm $R^2$")

        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month

        # ym_r2 = []
        # df_kelly['ym'] = df_kelly['date'].dt.year * 100 + df_kelly['date'].dt.month
        # for v in MODEL_ALL:
        #     tt = df_kelly.loc[~pd.isna(df_kelly[v]), :].groupby(['ym']).apply(lambda x: r2(x, v))
        #     tt.name = m_dict[v]
        #     ym_r2.append(tt)
        # ym_r2 = pd.DataFrame(ym_r2).T
        # ym_r2 = ym_r2.fillna(0.0)
        # # ym_r2.columns = [m_dict[c] for c in ym_r2.columns]
        # g = sns.pairplot(ym_r2)
        # g.map_lower(corrfunc)
        #
        # plt.tight_layout()
        # plt.savefig(paper.dir_figs + 'corr_month.png')
        #
        # self.plt_show()
        #
        # paper.append_fig_to_sec(fig_names=['corr_month'], sec_name='Results', size="80mm",
        #                         main_caption=r"The figures illustrate the correlation between each models month by month $R^2$")

        # ### add the new section with the
        # paper.append_text_to_sec('Results',r'\n \n \clearpage \n \n')
        # paper.create_new_sec('subsample_analysis')
        # paper.append_text_to_sec('subsample_analysis',r'\section{Subsample Analysis}')

        comp = self.model.data.load_compustat(True)
        comp_col = list(comp.columns[2:])
        df['year'] = df['date'].dt.year
        final = df_kelly.merge(comp)

        def get_expanding_r2(temp):
            R = []
            for y in tqdm(YEAR, 'compute R^2 expanding'):
                ind = temp['date'].dt.year <= y
                r = {'year': y}
                for c in MODEL_ALL:
                    ind_na = ~pd.isna(temp[c])
                    r[c] = r2(temp.loc[ind & ind_na, :], c)
                R.append(r)
            res = pd.DataFrame(R)
            res.index = res['year']
            del res['year']
            return res

        def plot_expanding_r2(res, y_min, y_max):
            # cummulative r2 plots
            d = pd.to_datetime(res.index, format='%Y')
            for i, c in enumerate(MODEL_ALL):
                plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=m_dict[c])
            plt.grid()
            # plt.xlabel('Year')
            plt.ylabel(r'$R^2_{oos}$')
            plt.ylim(y_min, y_max)
            plt.legend()
            plt.tight_layout()

        for col in comp_col:
            print('#########', col)
            final['q'] = pd.qcut(final[col], q=3, labels=False, duplicates='drop')
            low = get_expanding_r2(final.loc[final['q'] == 0, :].copy())
            high = get_expanding_r2(final.loc[final['q'] == 2, :].copy())

            y_min = min(low.min().min(), high.min().min()) * 1.1
            y_max = max(low.max().max(), high.max().max()) * 1.1

            plot_expanding_r2(low, y_min, y_max)
            plt.savefig(paper.dir_figs + f'{col}_low.png')
            self.plt_show()

            plot_expanding_r2(high, y_min, y_max)
            plt.savefig(paper.dir_figs + f'{col}_high.png')
            self.plt_show()

            tt = low.iloc[-1, :] - high.iloc[-1, :]
            s = r'$R^2$ diff: '
            for k in tt.index:
                s = s + k + f' {np.round(tt[k], 5)}'

            paper.append_fig_to_sec(fig_names=[f'{col}_low', f'{col}_high'], fig_captions=['low', 'high'], sec_name='Results',
                                    main_caption=rf"The figures compare the cumulative $R^2$ of the models splitting by {col}. "
                                                 rf"Panel (a) shows the lowest third {col}, while panel (b) shows the highest {col}. "
                                                 rf"Total {s}")

        ##################
        # creating sr table for us and kelly
        ##################
        weight_by = 'mkt_cap'
        strategy = Strategy(par)
        kelly = strategy.portfolio_quantile_sort(df_strat, Q=10, ret_col='ret1m', signal='NN4EW', w_col=weight_by)
        kelly_pred = strategy.portfolio_quantile_sort(df_strat, Q=10, ret_col='NN4EW', signal='NN4EW', w_col=weight_by)
        us = strategy.portfolio_quantile_sort(df_strat, Q=10, ret_col='ret1m', signal='pred', w_col=weight_by)
        us_pred = strategy.portfolio_quantile_sort(df_strat, Q=10, ret_col='pred', signal='pred', w_col=weight_by)

        us_table = strategy.table_quantile_mean_std_sharp_pred(us, us_pred, annualize=12)
        kelly_table = strategy.table_quantile_mean_std_sharp_pred(kelly, kelly_pred, annualize=12)
        us_table.to_latex(paper.dir_tables+'sharp_t_us.tex', multicolumn=True)
        kelly_table.to_latex(paper.dir_tables+'sharp_t_k.tex', multicolumn=True)

        paper.append_table_to_sec('sharp_t_us',sec_name='Results',caption='The table below shows the portfolio built on our prediction weighted by market-cap.')
        paper.append_table_to_sec('sharp_t_k',sec_name='Results',caption='The table below shows the portfolio built on the Kelly prediction weighted by market-cap.')

        ##################
        # PORTFOLIOS
        ##################

        ## creating protfoio old way
        def get_port_old_version(pred='pred', Q=5):
            df = full_df.copy()
            df = df.loc[~pd.isna(df[pred]), :]
            df = df.loc[~pd.isna(df['ret1m']), :]
            df['port'] = df.groupby('date')[pred].apply(lambda x: pd.qcut(x, Q, labels=False, duplicates='drop'))
            df = df.groupby(['port', 'date'])[name_ret].mean().reset_index()
            m = df.groupby('port').mean()
            s = df.groupby('port').std()

            for port in df['port'].unique():
                port = int(port)
                t = df.loc[df['port'] == port, :].copy()
                t['t'] = t['date'].diff().dt.days.fillna(0)
                t['tt'] = t['t'].cumsum()
                ind = t['tt'] % 60 == 0
                t = t.loc[ind, :]
                t.index = t['date']
                r = 1 + t[name_ret]
                plt.plot(r.index, r.cumprod(), label=f'Port {int(port)}', color=didi.DidiPlot.COLOR[port])
            plt.legend()
            # plt.xlabel('Date')
            plt.ylabel('Return')
            plt.grid()
            plt.tight_layout()
            plt.savefig(paper.dir_figs + f'cum_ret_{pred}.png')
            self.plt_show()
            m = m.rename(columns={name_ret: pred})
            s = s.rename(columns={name_ret: pred})
            return m, s

        M = []
        S = []

        for tp in MODEL_LIST:
            m, s = get_port_old_version(tp)
            M.append(m)
            S.append(s)

        M = pd.concat(M, 1)
        k = -1
        for tp in MODEL_LIST:
            k += 1
            plt.plot(M.index, M[tp], label=f'{tp}', color=didi.DidiPlot.COLOR[k])
            plt.scatter(M.index, M[tp], color=didi.DidiPlot.COLOR[k])
        plt.legend()
        plt.xlabel('Portfolio quintile')
        plt.ylabel('Portfolio mean return')
        plt.xticks(M.index)
        plt.grid()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'comparing_mean.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=[f'comparing_mean'], fig_captions=['Mean'], sec_name='Results',
                                main_caption=rf"The figure above show the mean return of the portfolio split in quintile based on the models predictions.")

        paper.append_fig_to_sec(fig_names=[f'cum_ret_{pred}' for pred in MODEL_LIST], fig_captions=MODEL_LIST, sec_name='Results',
                                main_caption=rf"The figures above show the cumulative return across time of the model sorted by pred-quintiles.")

        ### changing the weights

        def get_port_weights(pred='pred', LEVERAGE=1):
            df = full_df.copy()
            df = df.loc[~pd.isna(df[pred]), :]
            df = df.loc[~pd.isna(df['ret1m']), :]
            df['w'] = (1 + df[pred]) ** LEVERAGE

            df['ew'] = 1
            df['rw'] = df['w'] * df[name_ret]
            df = df.groupby('date')[['w', 'rw', name_ret, 'ew']].sum()
            df['r'] = df['rw'] / df['w']
            df['re'] = df[name_ret] / df['ew']

            t = df.copy()
            t['date'] = t.index
            t['t'] = t['date'].diff().dt.days.fillna(0)
            t['tt'] = t['t'].cumsum()
            ind = t['tt'] % 60 == 0
            t = t.loc[ind, :]

            def get_m(input='r'):
                m = df[input].agg(['mean', 'std'])
                m['mean'] = (1 + m['mean']) ** (252 / self.par.data.H) - 1
                m['std'] = m['std'] * np.sqrt(252 / self.par.data.H)
                m['sharp'] = m['mean'] / m['std']
                return m

            get_m('r')
            get_m('re')

            return get_m('r'), get_m('re'), t

        def get_all_weight_perf(pred='pred'):
            R = []
            for leverage in tqdm(np.arange(1, 41, 1), f'loop for all weight perf {pred}'):
                r, re, t = get_port_weights(pred=pred, LEVERAGE=leverage)
                r.name = leverage
                R.append(r)
            return pd.concat(R, 1).T

        wp = {}
        for tp in MODEL_LIST:
            t = get_all_weight_perf(tp)
            wp[tp] = t

        k = -1
        min_ = 0
        max_ = 100
        min_y = 0
        max_y = 100
        for tp in MODEL_LIST:
            k += 1
            df = wp[tp]
            min_ = max(min_, df['std'].min())
            max_ = min(max_, df['std'].max())
            min_y = max(min_y, df['mean'].min())
            max_y = min(max_y, df['mean'].max())
            plt.plot(df['std'], df['mean'], label=m_dict[tp], color=didi.DidiPlot.COLOR[k])

        plt.legend()
        plt.xlabel('annualized standard deviation')
        plt.ylabel('annualized mean return')
        plt.xlim(min_, max_)
        plt.ylim(min_y - 0.005, max_y + 0.005)
        plt.grid()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'mean_vs_factor.png')
        self.plt_show()

        k = -1
        for tp in MODEL_LIST:
            k += 1
            df = wp[tp]
            plt.plot(df.index, df['sharp'], label=f'{tp}', color=didi.DidiPlot.COLOR[k])

        plt.legend()
        plt.xlabel('Weight factor')
        plt.ylabel('weighted portfolio mean return')
        plt.grid()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'sharp_vs_factor.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=[f'mean_vs_factor', f'sharp_vs_factor'], fig_captions=['Mean', 'Sharp'], sec_name='Results',
                                main_caption=rf"The figures above show the mean vs std (panel a) and sharp-ratio (panel b) return of the portfolio built on weighted by a factor of $(1+r)^{{fact}}$, "
                                             rf"where r is the predcition and fact is the factor on the x-axis of panel b.")

        ##################
        # shapeley if callculated
        ##################
        L = [x for x in os.listdir(self.model.res_dir) if 'shap_' in x]
        shapely_exist = len(L) > 0
        if shapely_exist:
            shap = pd.DataFrame()
            for l in tqdm(L, 'load shapeley df'):
                shap = shap.append(pd.read_pickle(self.model.res_dir + l))

            ## renaming the columns to please the master

            def tr_func(x):
                t = ''
                if x == 'mean_pred':
                    t = 'mean'
                elif x == 'median_pred':
                    t = 'median'
                else:
                    nb_days = x.split('_')[-1]
                    predictor = x.split('err_')[1].split('_')[0]
                    if predictor == 'true':
                        predictor = 'return'
                    agg_id = x.split('err_' + predictor + '_')[-1].split('_')[0]
                    if agg_id == 'err':
                        agg_id = x.split('err_true_ret_')[-1].split('_')[0]
                    if 'mean' in agg_id:
                        agg = 'average absolute error'
                    elif 'std' in agg_id:
                        agg = 'variance absolute error'
                    else:
                        agg = 'return'
                    if 'Quantile' in agg_id:
                        if '0.75' in agg_id:
                            agg = 'upper quartile absolute error'
                        if '0.25' in agg_id:
                            agg = 'lower quartile absolute error'
                    t = f'{predictor} predictor | {nb_days} days {agg}'
                return t



            tr = []
            print(shap.columns)

            for x in shap.columns:
                if x in shap.columns[5:]:
                    tr.append(tr_func(x))
                else:
                    tr.append(x)
            shap.columns = tr

            ## keep only the day on the vilk sample
            t = full_df.dropna()[['permno', 'date', 'pred', 'ret1m']]
            shap = shap.merge(t[['permno', 'date']])
            f = r2(t, 'pred')
            S = {}

            for c in shap.columns[5:]:
                print(c)
                S[c] = (f / r2(shap, c)) - 1
                # S[c] = (r2(shap,c)-f)

            plt.figure(figsize=[6.4 * 2, 4.8 * 1.5])
            S = pd.Series(S).sort_values(ascending=True)
            plt.barh(S.index, S.values)
            plt.tight_layout()
            plt.savefig(paper.dir_figs + 'shap_clean.png')
            self.plt_show()

            sk = S[S > 0]
            sk /= sk.sum()
            if sk.shape[0] > 20:
                sk = sk.head(20)
            sk = sk.sort_values(ascending=True)

            plt.figure(figsize=[6.4 * 2, 4.8 * 1.5])
            plt.barh(sk.index, sk.values)
            plt.tight_layout()
            plt.savefig(paper.dir_figs + 'shap_kelly.png')
            self.plt_show()
            sk = S[S > 0]
            sk /= sk.sum()
            # for x in shap.columns[5:]:
            #     if '_20' in x:
            #         print(x)

            paper.append_fig_to_sec(fig_names=[f'shap_clean', f'shap_kelly'], fig_captions=['Clean', 'Kelly'], sec_name='Results',
                                    main_caption=rf"The figures above show the pseudo-shapely values kelly style. The first figure show the full results (panel a) while the second present the kelly subset (panel b) where we normalize the positive shapely values to 1.")

            ## shapeley clean year per year
            S_year = []
            for y in tqdm(YEAR, 'shapely year per year'):
                f = r2(t.loc[t['date'].dt.year == y, :], 'pred')
                S = {}
                for c in shap.columns[5:]:
                    S[c] = (f / r2(shap.loc[shap['date'].dt.year == y, :], c)) - 1
                    # S[c] = (r2(shap,c)-f)
                S = pd.Series(S).sort_values(ascending=True)
                print(y, S.head())
                plt.figure(figsize=[6.4 * 2, 4.8 * 1.5])
                plt.barh(S.index, S.values)
                plt.tight_layout()
                plt.savefig(paper.dir_figs + f'shap_clean_{y}.png')
                self.plt_show()
                S.name = y
                S_year.append(S)
            plt.figure(figsize=[6.4 * 2, 4.8 * 1.5])
            t = pd.concat(S_year, 1)
            S = t.max(1).sort_values(ascending=True)
            plt.barh(S.index, S.values)
            plt.tight_layout()
            plt.savefig(paper.dir_figs + 'shap_min.png')
            self.plt_show()

            paper.append_fig_to_sec(fig_names=[f'shap_min'], sec_name='Results',
                                    main_caption=rf"The figures above show the minimum shapely value caluclated year by year. It shows the maximum positive impact a feature had on a given year.")

            def get_big_fig(basis):
                to_plot = [x for x in t.index if basis in x]
                if 'return' not in basis:
                    min_ = t.loc[to_plot + [basis.split(' ')[0]],:].min().min()*0.95
                    max_ = t.loc[to_plot + [basis.split(' ')[0]],:].max().max()*1.05
                else:
                    min_ = t.loc[to_plot,:].min().min()*0.95
                    max_ = t.loc[to_plot,:].max().max()*1.05

                nb_plot = len(to_plot) + 1

                plt.figure(figsize=[6.4 * 3, 4.8 * 4])
                k = 0
                CC = ['average absolute error', 'lower quartile absolute error', 'upper quartile absolute error', 'variance absolute error']
                for cc in CC:
                    kk = -1
                    for d in [20, 180, 252]:
                        c = basis + f' {d} days {cc}'
                        k += 1
                        kk += 1
                        plt.subplot(len(CC) + 1, int(np.ceil(nb_plot) / len(CC)), k)
                        t.loc[c, :].plot(color='k')
                        plt.ylim(min_,max_)
                        plt.grid()
                        plt.tight_layout()
                        if k <= 3:
                            # plt.title(f'{d} days', fontweight='bold')
                            plt.title(f'{d} days')
                        if (k - 1) % (len(CC) - 1) == 0:
                            plt.ylabel(cc.lower())

                k += 1
                plt.subplot(len(CC) + 1, int(np.ceil(nb_plot) / len(CC)), k)
                t.loc[basis.split(' ')[0], :].plot(color='k')
                # plt.title(basis.split(' ')[0], fontweight='bold')
                plt.title(basis.split(' ')[0])
                plt.ylabel('forecast')
                plt.grid()
                plt.ylim(min_, max_)
                plt.tight_layout()


            get_big_fig('mean predictor |')
            plt.savefig(paper.dir_figs + f'big_mean.png')
            self.plt_show()

            # get_big_fig('median predictor |')
            # plt.savefig(paper.dir_figs + f'big_median.png')
            # self.plt_show()

            if self.par.data.include_mom:
                get_big_fig('return predictor |')
                plt.savefig(paper.dir_figs + f'big_return.png')
                self.plt_show()


            paper.append_fig_to_sec(fig_names='big_mean', sec_name='Results',
                                    main_caption=rf"The figures above show the time series of shapely value year per year with mean predictor.")


            paper.append_fig_to_sec(fig_names='big_median', sec_name='Results',
                                    main_caption=rf"The figures above show the time series of shapely value year per year with median predictor.")


            if self.par.data.include_mom:
                paper.append_fig_to_sec(fig_names='big_ret', sec_name='Results',
                                        main_caption=rf"The figures above show the time series of shapely value year per year with return predictor.")


    def plt_show(self):
        plt.close()

# self = Trainer()
# self.create_paper()
