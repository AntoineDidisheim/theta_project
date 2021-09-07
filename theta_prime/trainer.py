import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import didipack as didi
import os
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import shutil

par = Params()

class Trainer:
    def __init__(self, par = Params()):
        self.par = par
        # self.par.name = 'defaultL64_32_16_Lr001Dropout001BS512ActreluOutRange05LossMSERetLOGd3OptCompCrspEXT'
        self.model = NetworkMean(self.par)

    def launch_training_expanding_window(self):
        self.model.data.load_internally()
        YEAR = range(2000,2020)
        for year in YEAR:
            self.model.run_year(year)

    def create_paper(self):
        # define the paper maine directory (can be relative or abolute path)
        f_dir = self.par.model.tex_dir+'/'+str(self.par.name)+'/'
        if os.path.exists(f_dir):
            shutil.rmtree(f_dir)
        paper = didi.LatexPaper(dir_= f_dir)
        # Creating the paper itself with a given title, author name and some default sections (non-mandatory)
        paper.create_paper(self.par.model.tex_name, title="KiSS: Keep it Simple Stupid", author="Antoine Didisheim", sec=['Introduction','Results'])
        MODEL_LIST=['pred', 'vilk', 'mw']
        m_dict = {'pred':'NNET','vilk':'vilk','mw':'mw'}

        L = [x for x in os.listdir(self.model.res_dir) if 'perf_' in x]
        full_df = pd.DataFrame()
        for l in tqdm(L,'load original df'):
            full_df = full_df.append(pd.read_pickle(self.model.res_dir + l))

        ## add martin wagner
        mw = self.model.data.load_mw()
        full_df = full_df.merge(mw,how='left')

        def r2(df_, col='pred'):
            r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - 0.0) ** 2).sum()
            return r2_pred

        ##################
        # r2 plots
        ##################

        df = full_df.dropna().copy()
        YEAR = np.sort(df['date'].dt.year.unique())
        R = []
        for y in tqdm(YEAR,'compute R^2 expanding'):
            ind = df['date'].dt.year <= y
            r = {'year':y}
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
        for i, c in enumerate(['pred', 'vilk','mw']):
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
                                main_caption=r"The figures above show the $R^2$ of the main models ")

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


        paper.append_fig_to_sec(fig_names=['corr_firms'], sec_name='Results', size = "80mm",
                                main_caption=r"The figures illustrate the correlation between each models firm by firm $R^2$")


        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month

        ym_r2 =[]
        for v in MODEL_LIST:
            tt = df.groupby(['ym']).apply(lambda x: r2(x,v))
            tt.name = m_dict[v]
            ym_r2.append(tt)
        ym_r2=pd.DataFrame(ym_r2).T
        g = sns.pairplot(ym_r2)
        g.map_lower(corrfunc)
        
        plt.tight_layout()
        plt.savefig(paper.dir_figs + 'corr_month.png')
        
        self.plt_show()


        paper.append_fig_to_sec(fig_names=['corr_month'], sec_name='Results', size = "80mm",
                                main_caption=r"The figures illustrate the correlation between each models month by month $R^2$")

    def plt_show(self):
        plt.close()




# self = Trainer()
# self.create_paper()