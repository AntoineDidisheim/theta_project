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

        MODEL_LIST = ['pred', 'vilk', 'mw']
        m_dict = {'pred': 'NNET', 'vilk': 'vilk', 'mw': 'mw'}

        L = [x for x in os.listdir(self.model.res_dir) if 'perf_' in x]
        print('look for data in dir', self.model.res_dir)
        print('found', L)
        full_df = pd.DataFrame()
        for l in tqdm(L, 'load original df'):
            full_df = full_df.append(pd.read_pickle(self.model.res_dir + l))
        true_full = full_df.copy()

        ## add martin wagner
        mw = self.model.data.load_mw()
        full_df = full_df.merge(mw, how='left')
        full_df = full_df.loc[~pd.isna(full_df['mw']), :]
        full_df = full_df.loc[~pd.isna(full_df['vilk']), :]
        full_df = full_df.reset_index(drop=True)

        name_ret = self.name_ret

        def r2(df_, col='pred'):
            if self.par.data.H == 20:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - 0.0) ** 2).sum()
            else:
                r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - (1.06 ** (self.par.data.H / 252) - 1)) ** 2).sum()
                # r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - df_[name_ret]) ** 2).sum()
                # r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - df[name_ret].mean()) ** 2).sum()
                # r2_pred = 1 - ((df_[name_ret] - df_[col]) ** 2).sum() / ((df_[name_ret] - 0.0) ** 2).sum()
            return r2_pred

        # df[name_ret].mean()
        if self.par.data.H == 20:
            ADD_TEXT = r' $R^2$ is defined with 0.0 as a denominator benchmark. '
        else:
            ADD_TEXT = r' $R^2$ is defined with in sample mean as a denominator benchmark. '

        ##################
        # r2 comparing to kelly (FULL CROSS SECTION)
        ##################
        KELLY_MODEL_LIST = ['pred', 'RFEW', 'NN4EW', 'NN3EW']
        k = self.model.data.load_kelly_bench()
        df = true_full.merge(k)
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
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=c)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Cummulative $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'KELLY_cumulative_r2.png')
        self.plt_show()

        ##################
        # kelly benchmark smaller subsample
        ##################
        df = df.loc[~pd.isna(df['vilk']), :]

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
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=c)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Cummulative $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'KELLY_SMALL_cumulative_r2.png')
        self.plt_show()

        # We now add two figure side by side
        paper.append_fig_to_sec(fig_names=['KELLY_cumulative_r2', 'KELLY_SMALL_cumulative_r2'], sec_name='Results',
                                fig_captions=['Cumulative', 'Year per year'],  # you can add individual caption to each figure (leave empty for no label)
                                main_caption=r"The figures above show the $R^2$ of our model against some of the best model from Kelly's paper. "
                                             r"The first figure show the result on the large cross section of assets, the second only those present also in the vilknoy and MW papers. " + ADD_TEXT)

        ##################
        # r2 plots
        ##################

        df = full_df.dropna().copy()
        r2(df, 'vilk')

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
            plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=c)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Cummulative $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'cumulative_r2.png')
        self.plt_show()

        # year per year
        df['year'] = df['date'].dt.year
        for i, c in enumerate(['pred', 'vilk', 'mw']):
            t = df.groupby('year').apply(lambda x: r2(x, col=c))
            plt.plot(d, t, color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=c if c != 'pred' else 'NNET')
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Year per Year $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'year_r2.png')
        self.plt_show()

        # We now add two figure side by side
        paper.append_fig_to_sec(fig_names=['cumulative_r2', 'year_r2'], sec_name='Results',
                                fig_captions=['Cumulative', 'Year per year'],  # you can add individual caption to each figure (leave empty for no label)
                                main_caption=r"The figures above show the $R^2$ of the main models. " + ADD_TEXT)

        tt = df.groupby('year')[['pred', 'mw', 'vilk']].std()
        plt.plot(tt.index, tt['pred'], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0], label='us')
        plt.plot(tt.index, tt['mw'], color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1], label='MW')
        plt.plot(tt.index, tt['vilk'], color=didi.DidiPlot.COLOR[2], linestyle=didi.DidiPlot.LINE_STYLE[2], label='vilk')
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Pred. std')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'const_pred.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['const_pred'], sec_name='Results',

                                main_caption=r"The figure above show the standard deviation year per year of the predicitons. It is here to check constant predicitons. ")

        ##################
        # full sample perf
        ##################
        YEAR = np.sort(true_full['date'].dt.year.unique())
        R_full = []
        for y in tqdm(YEAR, 'compute R^2 expanding FULL'):
            ind = df['date'].dt.year <= y
            r = {'year': y}
            r['pred'] = r2(true_full.loc[ind, :], 'pred')

            R_full.append(r)
        res_full = pd.DataFrame(R_full)
        res_full.index = res_full['year']
        del res_full['year']

        # cummulative r2 plots
        d = pd.to_datetime(res_full.index, format='%Y')
        plt.plot(d, res_full['pred'], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0], label='perf full sample')
        plt.plot(d, res['pred'], color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1], label='perf vilk sample')
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel(r'Cummulative $R^2$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'cumulative_full_sample.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['cumulative_full_sample'], sec_name='Results',
                                main_caption=r"The figures above compare our model's performance on the full sample versus the vilknoy's subsmaple. Both line show the $R^2$ computed on an expanding window.")

        del true_full

        ##################
        # stock by stock
        ##################
        S = []
        df['year'] = df['date'].dt.year
        for i, c in enumerate(MODEL_LIST):
            t = df.groupby('permno').apply(lambda x: r2(x, col=c))
            plt.hist(t, color=didi.DidiPlot.COLOR[i], alpha=0.5, label=c if c != 'pred' else 'NNET', bins=50, density=True)
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

        ### violin noow :)
        violin = pd.DataFrame()
        for v in MODEL_LIST:
            t1 = t[[v]].copy().rename(columns={v: r'$R^2$ per firm'})
            t1['model'] = m_dict[v]
            violin = violin.append(t1)
        del t1

        mkt = self.model.data.load_additional_crsp()
        mkt = mkt.groupby('permno').mean().reset_index()
        mkt['Market cap. Quintile'] = pd.qcut(mkt['mkt_cap'], 5, labels=False, duplicates='drop')
        violin = violin.reset_index().merge(mkt)

        sns.violinplot(data=violin, x='Market cap. Quintile', y=r'$R^2$ per firm', hue='model', palette='muted')
        plt.savefig(paper.dir_figs + 'SIZE_violin_stock_r2.png')
        self.plt_show()

        sns.boxplot(data=violin, x='Market cap. Quintile', y=r'$R^2$ per firm', hue='model', palette='muted')
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'SIZE_box_stock_r2.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['SIZE_violin_stock_r2', 'SIZE_box_stock_r2'], sec_name='Results',
                                main_caption=r"The figures compare the $R^2$ firm by firm sorted by market cap. size of each model with violin plots (a) and boxplots (b). "
                                             r"For both figure, we compute the average market size across the sample to determine the size quintile of the firm.")

        ##################
        # correlation of perf
        ##################

        def corrfunc(x, y, ax=None, **kws):
            """Plot the correlation coefficient in the top left hand corner of a plot."""
            r, _ = pearsonr(x, y)
            ax = ax or plt.gca()
            ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

        g = sns.pairplot(t)
        g.map_lower(corrfunc)
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'corr_firms.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=['corr_firms'], sec_name='Results', size="80mm",
                                main_caption=r"The figures illustrate the correlation between each models firm by firm $R^2$")

        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month

        ym_r2 = []
        for v in MODEL_LIST:
            tt = df.groupby(['ym']).apply(lambda x: r2(x, v))
            tt.name = m_dict[v]
            ym_r2.append(tt)
        ym_r2 = pd.DataFrame(ym_r2).T
        g = sns.pairplot(ym_r2)
        g.map_lower(corrfunc)

        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'corr_month.png')

        self.plt_show()

        paper.append_fig_to_sec(fig_names=['corr_month'], sec_name='Results', size="80mm",
                                main_caption=r"The figures illustrate the correlation between each models month by month $R^2$")

        # ### add the new section with the
        # paper.append_text_to_sec('Results',r'\n \n \clearpage \n \n')
        # paper.create_new_sec('subsample_analysis')
        # paper.append_text_to_sec('subsample_analysis',r'\section{Subsample Analysis}')

        comp = self.model.data.load_compustat(True)
        comp_col = list(comp.columns[2:])
        df['year'] = df['date'].dt.year
        final = df.merge(comp)

        def get_expanding_r2(temp):
            R = []
            for y in tqdm(YEAR, 'compute R^2 expanding'):
                ind = temp['date'].dt.year <= y
                r = {'year': y}
                for c in MODEL_LIST:
                    r[c] = r2(temp.loc[ind, :], c)
                R.append(r)
            res = pd.DataFrame(R)
            res.index = res['year']
            del res['year']
            return res

        def plot_expanding_r2(res, y_min, y_max):
            # cummulative r2 plots
            d = pd.to_datetime(res.index, format='%Y')
            for i, c in enumerate(MODEL_LIST):
                plt.plot(d, res[c], color=didi.DidiPlot.COLOR[i], linestyle=didi.DidiPlot.LINE_STYLE[i], label=c)
            plt.grid()
            plt.xlabel('Year')
            plt.ylabel(r'Cummulative $R^2$')
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
        # PORTFOLIOS
        ##################

        ## creating protfoio old way
        def get_port_old_version(pred='pred', Q=5):

            df = full_df.copy()
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
            plt.xlabel('Date')
            plt.ylabel('Cum. Ret.')
            plt.grid()
            plt.tight_layout()
            plt.savefig(paper.dir_figs + f'cum_ret_{pred}.png')
            self.plt_show()
            m = m.rename(columns={name_ret: pred})
            s = s.rename(columns={name_ret: pred})
            return m, s

        M = []
        S = []
        TP = ['mw', 'vilk', 'pred']
        for tp in TP:
            m, s = get_port_old_version(tp)
            M.append(m)
            S.append(s)

        M = pd.concat(M, 1)
        k = -1
        for tp in TP:
            k += 1
            plt.plot(M.index, M[tp], label=f'{tp}', color=didi.DidiPlot.COLOR[k])
        plt.legend()
        plt.xlabel('Portfolio index')
        plt.ylabel('Port mean return')
        plt.grid()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'comparing_mean.png')
        self.plt_show()

        paper.append_fig_to_sec(fig_names=[f'comparing_mean'], fig_captions=['Mean'], sec_name='Results',
                                main_caption=rf"The figure above show the mean return of the portfolio split in quintile based on the models predictions.")

        paper.append_fig_to_sec(fig_names=[f'cum_ret_{pred}' for pred in TP], fig_captions=TP, sec_name='Results',
                                main_caption=rf"The figures above show the cumulative return across time of the model sorted by pred-quintiles.")

        ### changing the weights

        def get_port_weights(pred='pred', LEVERAGE=1):
            df = full_df.copy()
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
            for leverage in tqdm(np.arange(1, 31, 1), f'loop for all weight perf {pred}'):
                r, re, t = get_port_weights(pred=pred, LEVERAGE=leverage)
                r.name = leverage
                R.append(r)
            return pd.concat(R, 1).T

        wp = {}
        for tp in TP:
            t = get_all_weight_perf(tp)
            wp[tp] = t

        k = -1
        for tp in TP:
            k += 1
            df = wp[tp]
            plt.plot(df['std'], df['mean'], label=f'{tp}', color=didi.DidiPlot.COLOR[k])

        plt.legend()
        plt.xlabel('Portfolio std')
        plt.ylabel('Portfolio mean')
        plt.grid()
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'mean_vs_factor.png')
        self.plt_show()

        k = -1
        for tp in TP:
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

            ## keep only the day on the vilk sample
            t = full_df.dropna()[['permno', 'date', 'pred', 'ret1m']]
            shap = shap.merge(t[['permno', 'date']])
            f = r2(t, 'pred')
            S = {}
            for c in shap.columns[5:]:
                S[c] = (f / r2(shap, c)) - 1
                # S[c] = (r2(shap,c)-f)
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
                plt.barh(S.index, S.values)
                plt.tight_layout()
                plt.savefig(paper.dir_figs + f'shap_clean_{y}.png')
                self.plt_show()
                S.name = y
                S_year.append(S)
            t = pd.concat(S_year, 1)
            S = t.max(1).sort_values(ascending=True)
            plt.barh(S.index, S.values)
            plt.tight_layout()
            plt.savefig(paper.dir_figs + 'shap_min.png')
            self.plt_show()

            paper.append_fig_to_sec(fig_names=[f'shap_min'], sec_name='Results',
                                    main_caption=rf"The figures above show the minimum shapely value caluclated year by year. It shows the maximum positive impact a feature had on a given year.")

            N = []
            L = []
            for c in t.index:
                t.loc[c,:].plot(color='k')
                plt.tight_layout()
                plt.title(c)
                plt.savefig(paper.dir_figs + f'shape_ts_{c}.png')
                self.plt_show()
                N.append(f'shape_ts_{c}')
                L.append(c.replace('_',' '))

            paper.append_fig_to_sec(fig_names=N,fig_captions=L, sec_name='Results', size = r"0.15\linewidth",
                                    main_caption=rf"The figures above show the time series of shapely value year per year .")


    def plt_show(self):
        plt.close()

# self = Trainer()
# self.create_paper()
