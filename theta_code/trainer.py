import pandas as pd
import numpy as np
from parameters import *
from ml_model import *
from data import *
import os
from matplotlib import pyplot as plt
import didipack as didi
import seaborn as sns


##################
# Set parameters
##################
# par = Params()
# # par.model.layers = [64,32,16]
# par.model.layers = [10]
# par.model.activation = 'sigmoid'
# par.model.batch_size = 32
# par.model.learning_rate = 0.001
# par.model.E = 10
# par.data.val_split = 0.1
# res = []
# par.update_model_name()

class Trainer:
    def __init__(self, par: Params):
        self.par = par
        self.paper = didi.LatexPaper(dir_=self.par.model.tex_dir)

        self.dir_tables = self.paper.dir_tables + par.name + '/'
        if not os.path.exists(self.dir_tables):
            os.makedirs(self.dir_tables)

        self.dir_figs = self.paper.dir_figs + par.name + '/'
        if not os.path.exists(self.dir_figs):
            os.makedirs(self.dir_figs)

        self.res_dir = 'res/' + par.name + '/'
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

    def plt_show(self):
        plt.close()


    def create_paper(self):
        self.paper.create_paper(self.par.model.tex_name, sec=['Results'], author="Antoine Didisheim, Fabio Trojani, Simon Scheidegger", title="Theta Project")

    def cv_training(self):
        model = NetworkTheta(self.par)
        res = []
        if self.par.model.cv in [CrossValidation.YEAR_BY_YEAR]:
            r_ = np.sort(model.data.label_df['date'].dt.year.unique()).tolist()
        if self.par.model.cv in [CrossValidation.EXPANDING]:
            r_ = np.sort(model.data.label_df['date'].dt.year.unique())[1:].tolist()
        if self.par.model.cv == CrossValidation.RANDOM:
            r_ = range(10)
        print('THE RANGE', r_)
        for i in r_:
            if self.par.model.cv in [CrossValidation.YEAR_BY_YEAR, CrossValidation.EXPANDING]:
                model.data.set_year_test(i)
            if self.par.model.cv == CrossValidation.RANDOM:
                model.data.move_shuffle()



            # r2, theta, mse, p_log, p_norm
            r2, theta, mse, p_log, p_norm = model.get_perf_oos()
            model.train()
            r2_new, theta_new, mse_new, p_log_new, p_norm_new = model.get_perf_oos()
            p_bench, r2_bench, mse_bench = model.get_bench_perf()

            r = model.data.test_label_df.copy()
            r['pred_no_train_log'] = p_log.numpy()
            r['pred_no_train_norm'] = p_norm.numpy()
            r['pred_log'] = p_log_new.numpy()
            r['pred_norm'] = p_norm_new.numpy()
            r['bench'] = p_bench.numpy()
            r['theta'] = theta_new
            res.append(r)
            print('########### r2')
            print('old', r2, 'new', r2_new, 'bench', r2_bench)
            print('########### mse')
            print('old', mse, 'new', mse_new, 'bench', mse_bench)
            model.create_network()
            ## resevae after each yerar
            print('#'*50)
            print('save year', r_)
            print('#'*50)
            pd.concat(res).to_pickle(self.res_dir + 'df.p')

    def create_report_sec(self):
        par = self.par
        with open(self.par.model.tex_dir + "/sec/Results.tex", mode='r') as file:
            all_of_it = file.read()
        if not "\input{sec/" + par.name + ".tex}" in all_of_it:
            # if section not already in results, we add it.
            self.paper.append_text_to_sec(sec_name="Results", text=r"\input{sec/" + par.name + ".tex}" + '\n' + '\n')

        self.paper.create_new_sec(sec_name=par.name)
        self.paper.append_text_to_sec(sec_name=par.name, text="\\clearpage \n \n")
        self.paper.append_text_to_sec(sec_name=par.name, text=r"\subsection{" + self.par.name.replace('_', ' ').replace('.','')+"}")

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
                                                                                                                                                                                                                                                                                  rf"\item Output min value {self.par.model.out_min}" + '\n' \
                                                                                                                                                                                                                                                                                                                                        rf"\item Batch size {self.par.model.batch_size}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                           rf"\item Option input {self.par.data.opt}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                        rf"\item Compustat input {self.par.data.comp}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         rf"\item CRSP input {self.par.data.crsp}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     rf"\item ret max: {self.par.data.max_ret}, min {self.par.data.min_ret}" + '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               r"\end{enumerate}" + '\n'


        self.paper.append_text_to_sec(sec_name=par.name, text=model_description.replace('_', ' '))

        df = pd.read_pickle(self.res_dir + 'df.p')




        # for now the return are all normal in the perf report
        ret_type_str = 'R'
        df['pred'] = df['pred_norm'].clip(-0.2,0.2)

        ### cross sectional trimming
        df['pred_abs'] = df['pred'].abs()
        df['max']=df.groupby('date')['pred_abs'].transform('quantile',0.99)
        df.loc[df['pred_abs'] > df['max'], 'pred'] = np.sign(df.loc[df['pred_abs'] > df['max'], 'pred']) * df.loc[df['pred_abs'] > df['max'], 'max']

        df['bench'] = df['bench'].clip(-0.2,0.2)




        # if self.par.data.ret == ReturnType.LOG:
        #     ret_type_str = 'log(R)'
        # else:
        #     ret_type_str  ='R'


        df['error_bench'] = (df['ret'] - df['bench']).abs()
        df['error_pred'] = (df['ret'] - df['pred']).abs()
        df.describe(np.arange(0,1.05,0.05)).round(3)





        ##################
        # multiple r2
        ##################
        ## add marting wagner
        # t=Data(self.par).marting_wagner_return()
        id_key = 'permno'
        mw = Data(self.par).marting_wagner_return()
        try:
            # if gvkey is main key, add permno to mw
            pr = Data(self.par).load_all_price()[['permno', 'gvkey']]
            pr['permno'] = pr['permno'].astype(int)
            pr = pr.drop_duplicates()
            mw = mw.merge(pr, how='left')
            id_key='gvkey'
        except:
            pass
        # their MW
        them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
        them['date'] = pd.to_datetime(them['date'])
        them['permno'] = them['permno'].astype(int)
        mw = mw.merge(them, how='left')
        t=mw[['date',id_key,'MW','mw30']]
        df = df.merge(t, how='left')
        df['mw30'] = df['mw30']/12
        # df['mw30'] = df['mw30']*20/30
        # their lower bound
        them = pd.read_csv(f'{self.par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})

        them['date'] = pd.to_datetime(them['date'])
        them['permno'] = them['permno'].astype(int)
        try:
            # again add permno if main is gvkey
            them = them.merge(pr)
        except:
            pass

        t=them[['date',id_key,'glb2_D30','glb3_D30']]
        df = df.merge(t, how='left')
        df['glb2_D30'] = df['glb2_D30'] / 12
        df['glb3_D30'] = df['glb3_D30'] / 12
        # df['glb3_D30'] = df['glb3_D30'] *20/30





        df['month'] = df['date'].dt.year*100+df['date'].dt.month



        try:
            t=Data(self.par).historical_theta()
            df = df.merge(t[['date',id_key,'pred']].rename(columns={'pred':'hist_theta'}), how='left')
            t = Data(self.par).load_all_price(False)
            t['year'] = t['date'].dt.year
            df['year'] =  df['date'].dt.year
            t=t.groupby('year')['ret'].mean().reset_index()
            t[r'$\bar{MKT}_{t-1}$']=t['ret'].shift()
            t = t.rename(columns={'ret':r'$\bar{MKT}_{t}$'})
            overall_average =  Data(self.par).load_all_price()['ret'].mean()
            df=df.merge(t)
        except:
            df[r'$\bar{MKT}_{t-1}$'] = np.nan
            df[r'hist_theta'] = np.nan
            overall_average = Data(self.par).load_all_price()['ret'].mean()

        try:
            t = Data(self.par)
            t.load_final()
            t.label_df[r'$\beta_{i,t} \bar{MKT}_{t}$'] = (t.p_df['beta_monthly']*t.p_df['mkt-rf'])/100
            df=df.merge(t.label_df)
        except:
            df[r'$\beta_{i,t} \bar{MKT}_{t}$'] = np.nan



        if 'hist_theta' not in df.columns:
            df['hist_theta']=np.nan
        if '$\\bar{MKT}_{t}$' not in df.columns:
            df['$\\bar{MKT}_{t}$']=np.nan
        df.describe(np.arange(0,1.05,0.05))
        ind = (df['ret']>=-0.5) & (df['ret']<=0.5)
        df = df.loc[ind,:]



        def r2(df_,y_bar, name='NNET'):
            try:
                if np.sum(pd.isna(y_bar))>0:
                    df_ = df_.loc[~pd.isna(y_bar),:]

                r2_pred = 1 - ((df_['ret'] - df_['pred']) ** 2).sum() / ((df_['ret'] - y_bar) ** 2).sum()
                r = (pd.Series({name: r2_pred})*100).round(2)
            except:
                r = np.nan
            return r
        def get_all_r(df):
            r=[
                r2(df,(1.06)**(1/12)-1,r'6\% premium'),
                r2(df,df['MW'],'Martin Wagner'),
                r2(df,df['mw30'],'Martin Wagner downloaded'),
                r2(df,0.0, r'$R=0.0$'),
                r2(df, df['hist_theta'],r'historical $\theta$'),
                r2(df, df['bench'],r'$\theta=1.0$'),
                r2(df, df[r'$\bar{MKT}_{t}$'],r'$\bar{MKT}_{t}$'),
                r2(df, df[r'$\bar{MKT}_{t-1}$'],r'$\bar{MKT}_{t-1}$'),
                r2(df, overall_average,r'$\bar{MKT}$'),
                r2(df, df[r'$\beta_{i,t} \bar{MKT}_{t}$'],r'$\beta_{i,t} \bar{MKT}_{t}$'),
                r2(df, df[r'glb2_D30'],r'Vilkny glb2 D30'),
                r2(df, df[r'glb3_D30'],r'Vilkny glb3 D30')
            ]
            return pd.concat(r).sort_values()

        def r2_abs(df_,y_bar, name='NNET'):
            try:
                if np.sum(pd.isna(y_bar))>0:
                    df_ = df_.loc[~pd.isna(y_bar),:]

                r2_pred = 1 - ((df_['ret'] - df_['pred']).abs()).sum() / ((df_['ret'] - y_bar).abs()).sum()
                r = (pd.Series({name: r2_pred})*100).round(2)
            except:
                r = np.nan
            return r
        def get_all_abs(df):
            r=[
                r2_abs(df,(1.06)**(1/12)-1,r'6\% premium'),
                r2_abs(df,df['MW'],'Martin Wagner'),
                r2_abs(df,df['mw30'],'Martin Wagner downloaded'),
                r2_abs(df,0.0, r'$R=0.0$'),
                r2_abs(df, df['hist_theta'],r'historical $\theta$'),
                r2_abs(df, df['bench'],r'$\theta=1.0$'),
                r2_abs(df, df[r'$\bar{MKT}_{t}$'],r'$\bar{MKT}_{t}$'),
                r2_abs(df, df[r'$\bar{MKT}_{t-1}$'],r'$\bar{MKT}_{t-1}$'),
                r2_abs(df, overall_average,r'$\bar{MKT}$'),
                r2_abs(df, df[r'$\beta_{i,t} \bar{MKT}_{t}$'],r'$\beta_{i,t} \bar{MKT}_{t}$'),
                r2_abs(df, df[r'glb2_D30'],r'Vilkny glb2 D30'),
                r2_abs(df, df[r'glb3_D30'],r'Vilkny glb3 D30')
            ]
            return pd.concat(r).sort_values()

        df['year'] = df['date'].dt.year


        t_abs=df.groupby('year').apply(lambda x: get_all_abs(x)).reset_index()
        t_abs.columns = ['Year','Type',r'$R^2$']
        t_abs=t_abs.pivot(columns='Year',index='Type')
        t_abs['All'] = get_all_abs(df)
        t_abs=t_abs.sort_values('All')
        tt = df.groupby('year')['date'].count()
        tt['All'] = df.shape[0]
        t_abs  = t_abs.T
        t_abs['nb. obs'] = tt.values
        t_abs = t_abs.T


        t=df.groupby('year').apply(lambda x: get_all_r(x)).reset_index()
        t.columns = ['Year','Type',r'$R^2$']
        t=t.pivot(columns='Year',index='Type')
        t['All'] = get_all_r(df)
        t=t.sort_values('All')
        tt = df.groupby('year')['date'].count()
        tt['All'] = df.shape[0]
        t  = t.T
        t['nb. obs'] = tt.values
        t = t.T

        # ### r2 with cheat
        # df_t = df.copy()
        # df_t['weird']=df_t.groupby(['year','glb2_D30'])['glb2_D30'].transform('count')
        # ind=(df_t['weird']>20) & (df_t['pred']>df_t['glb2_D30'])
        # df_t.loc[ind,'pred']  =df_t.loc[ind,'glb2_D30']
        #
        # t = df_t.groupby('year').apply(lambda x: get_all_r(x)).reset_index()
        # t.columns = ['Year', 'Type', r'$R^2$']
        # t = t.pivot(columns='Year', index='Type')
        # t['All'] = get_all_r(df)
        # t = t.sort_values('All')
        # tt = df.groupby('year')['date'].count()
        # tt['All'] = df.shape[0]
        # t = t.T
        # t['nb. obs'] = tt.values
        # t = t.T

        t.to_latex(self.dir_tables+'all_r2.tex', escape=False)
        self.paper.append_table_to_sec(table_name='all_r2.tex', resize=0.95, sec_name=par.name, sub_dir=par.name,
                                       caption='The table below shows the out of sample $L_2 R^2$ of our model against various benchmark')

        t_abs.to_latex(self.dir_tables+'all_r2_ABS.tex', escape=False)
        self.paper.append_table_to_sec(table_name='all_r2_ABS.tex', resize=0.95, sec_name=par.name, sub_dir=par.name,
                                       caption='The table below shows the out of sample $L_1 R^2$ of our model against various benchmark')



        ##################
        # beta analysis
        ##################
        t=df.groupby('date')[['pred','glb2_D30','ret']].mean().reset_index()
        t['year'] = t['date'].dt.year
        t['us_error'] = (t['pred'] - t['ret']) ** 2
        t['vilkny_error'] = (t['glb2_D30'] - t['ret']) ** 2
        t.groupby(['year'])['us_error','vilkny_error'].mean()
        t['alpha'] = 1.0

        max_lag = 4
        table = didi.TableReg(rename_dict={'pred':r'$\phi(X_i)$', 'glb2_D30':"Vilknoy's prediction"})
        m=sm.OLS(t['ret'],t[['pred','alpha']]).fit(cov_type='HAC',cov_kwds={'maxlags':max_lag})
        table.add_reg(m)
        m=sm.OLS(t['ret'],t[['glb2_D30','alpha']]).fit(cov_type='HAC',cov_kwds={'maxlags':max_lag})
        table.add_reg(m)
        m=sm.OLS(t['ret'],t[['pred']]).fit(cov_type='HAC',cov_kwds={'maxlags':max_lag})
        table.add_reg(m)
        m=sm.OLS(t['ret'],t[['glb2_D30']]).fit(cov_type='HAC',cov_kwds={'maxlags':max_lag})
        table.add_reg(m)
        m=sm.OLS(t['ret'],t[['glb2_D30','pred']]).fit(cov_type='HAC',cov_kwds={'maxlags':max_lag})
        table.add_reg(m)
        table.save_tex(self.dir_tables+'market_reg.tex')


        self.paper.append_table_to_sec(table_name='market_reg.tex', resize=0.95, sec_name=par.name, sub_dir=par.name,
                                       caption='The table above show result of regressions of average monthly returns on the average predictions')


        ##################
        # year year analysis
        ##################
        r2_list=t.loc['Vilkny glb2 D30',:].iloc[:-1].reset_index().sort_values('Vilkny glb2 D30')
        ind_2 = df.loc[:, 'pred'].abs() <= 0.05
        for y in r2_list['Year']:
            r2_v = r2_list.loc[r2_list['Year']==y,'Vilkny glb2 D30'].iloc[0]
            ind=df['year']==y
            ind_strong=ind & ind_2
            plt.scatter(df.loc[ind,'glb2_D30'],df.loc[ind,'pred'],marker='+',color='k')
            plt.ylabel('NNET')
            plt.xlabel('Glb2_D30')
            plt.tight_layout()
            plt.savefig(self.dir_figs + f'{y}_expl_weak.png')
            plt.close()

            plt.scatter(df.loc[ind_strong, 'glb2_D30'], df.loc[ind_strong, 'pred'], marker='+', color='k')
            plt.ylabel('NNET')
            plt.xlabel('Glb2_D30')
            plt.tight_layout()
            plt.savefig(self.dir_figs + f'{y}_expl_strong.png')
            plt.close()

            get_all_r(df.loc[ind&ind_2,:])


            self.paper.append_fig_to_sec(fig_names=[f'{y}_expl_weak', f'{y}_expl_strong'], sec_name=par.name, sub_dir=par.name, overall_label=f"yy_{y}",
                                         main_caption=fr"The figure above explore the relationship between our prediction and Vilkny's for year {y}. "
                                                      fr"Panel (a) show the full scatter plots, panel (b) cuts extreme prediction of the network at +-5\%."
                                                      fr"The overall Glb2 D30 $R^2$ for this year was {r2_v}")



        ##################
        # r2 plots
        ##################

        def r2(df_):
            r2_pred = 1 - ((df_['ret'] - df_['pred']) ** 2).sum() / ((df_['ret'] - 0.0) ** 2).sum()
            r2_bench = 1 - ((df_['ret'] - df_['bench']) ** 2).sum() / ((df_['ret'] - 0.0) ** 2).sum()
            return pd.Series({'NNET': r2_pred, rf'$\theta={self.par.model.bench_theta}$': r2_bench})

        r2_all = (r2(df) * 100).round(2)

        df['year'] = df['date'].dt.year
        yy = df.groupby('year').apply(lambda x: r2(x))
        plt.plot(yy.index, yy.iloc[:, 0]*100, label=yy.columns[0], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(yy.index, yy.iloc[:, 1]*100, label=yy.columns[1], color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(r'$R^2$')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'r2_ts.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names="r2_ts", sec_name=par.name, sub_dir=par.name,
                                     main_caption=fr"The figure above shows out of sample $R^2$ multiplied by 100 of the neural network and $\theta=1$ benchmark for year in the sample."
                                                  fr"The overal $R^2$ multiplied by 100 is equal to {r2_all.iloc[0]} for the NNET, and {r2_all.iloc[1]} for the benchmark.")

        ##################
        # prediction mean and quartiles
        ##################

        t = df.groupby('date')[['pred', 'bench']].mean().rolling(12).mean()
        k = 0
        plt.plot(t.index, t['pred'], label='NNET, mean', color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.plot(t.index, t['bench'], label=r'$\theta=1$, mean', color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.legend()
        plt.grid()
        plt.xlabel('Date')
        plt.ylabel(f'Predicted {ret_type_str}')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'mean_pred.png')
        self.plt_show()

        k = -1
        for q in [0.25, 0.5, 0.75]:
            t = df.groupby('date')[['pred', 'bench']].quantile(q).rolling(12).mean()
            k += 1
            if self.par.data.ret == ReturnType.LOG:
                pred = t['pred']*12
                bench = t['bench']*12
            if self.par.data.ret == ReturnType.RET:
                pred = (t['pred']+1)**12-1
                bench = (t['bench']+1)**12-1


            plt.plot(t.index, pred, label=rf'NNET, q={q}', color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.plot(t.index, bench, label=rf'$\theta=1$, q={q}', color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.grid()
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel(f'Predicted {ret_type_str}')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'q_pred.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=['mean_pred', 'q_pred'], sec_name=par.name, sub_dir=par.name, overall_label="forecast",
                                     main_caption=fr"The figure above show the forecast of the NNET and benchmark across time. "
                                                  fr"Panel (a) shows the mean predicted log-return while panel (b) shows the cross-sectional median and quartiles. "
                                                  fr"We smooth the time series with a 12 month moving average.")

        ##################
        # MSE acrosss time
        ##################
        df['mse_bench'] = ((df['ret'] - df['bench']) ** 2)
        df['mse_net'] = ((df['ret'] - df['pred']) ** 2)

        mse_bench = np.round(df['mse_bench'].mean(), 5)
        mse_net = np.round(df['mse_net'].mean(), 5)

        yy = df.groupby('date')[['mse_net', 'mse_bench']].mean().rolling(12).mean()
        yy['dMse'] = (yy['mse_net'] - yy['mse_bench']) / yy['mse_bench']
        yy.columns = ['NNET', r'$\theta=1.0$', 'dMse']
        plt.plot(yy.index, yy.iloc[:, 0], label=yy.columns[0], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(yy.index, yy.iloc[:, 1], label=yy.columns[1], color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(r'MSE')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'mse_ts.png')
        self.plt_show()

        plt.plot(yy.index, yy.iloc[:, 2], label=yy.columns[0], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.grid(True)
        plt.ylabel(r'$(MSE_{net}-MSE_{\theta=1.0})/MSE_{\theta=1.0}$')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'dmse_ts.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=["mse_ts", "dmse_ts"], sec_name=par.name, sub_dir=par.name, overall_label="mse",
                                     main_caption=fr"The figure above shows out of sample MSE of the neural network and $\theta=1$ benchmark for year in the sample. "
                                                  r"Panel (a) shows the average MSE of both models while panel (b) shows the relative error expressed in percentage of the benchmark error ($(MSE_{net}-MSE_{\theta=1.0})/MSE_{\theta=1.0}$). "
                                                  r"We smooth all time series with a 12 months moving average. "
                                                  fr"The average MSE on the full sample is equal to {mse_net} for the NNET, and {mse_bench} for the benchmark.")

        ##################
        # theta distribution
        ##################

        plt.hist(df['theta'], bins=100, color=didi.DidiPlot.COLOR[0])
        plt.xlabel(r'$\theta$')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'theta_hist.png')
        self.plt_show()

        yy = df.groupby('year')['pred'].count()
        yy=np.array(list(yy[yy>=200].index))
        N = []
        # col_nb=3
        # ly = np.ceil(len(yy)/col_nb)
        # plt.figure(figsize=(col_nb*6.4,ly*4.8))
        for i in range(1, len(yy)+1):
            # plt.subplot(ly,col_nb,i)
            ind = df['year']==yy[i-1]
            plt.hist(df.loc[ind,'theta'], bins=100, color=didi.DidiPlot.COLOR[0])
            # plt.xlabel(yy[i-1])
            plt.xlabel(r'$\theta$')
            plt.xlim(df['theta'].min(),df['theta'].max())
            plt.tight_layout()
            plt.savefig(self.dir_figs + f'theta_hist_{yy[i-1]}.png')
            N.append(f'theta_hist_{yy[i-1]}')
            self.plt_show()


        self.paper.append_fig_to_sec(fig_names=N, sec_name=par.name, sub_dir=par.name,fig_captions=[str(x) for x in yy],
                                     size=r'0.3\linewidth',
                                     main_caption=r"The figures above show the distribution of the predicted $\theta$ split year per year. "
                                                  r"We only show years with at least 200 observations")




        k = 0

        t = df.groupby('date')[['theta']].mean().rolling(12).mean()
        plt.plot(t.index, t['theta'], label=rf'Average $\theta$', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])

        for q in [0.25, 0.5, 0.75]:
            t = df.groupby('date')[['theta']].quantile(q).rolling(12).median()
            k += 1
            plt.plot(t.index, t['theta'], label=rf'$\theta$, q={q}', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.grid()
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel(f'Predicted {ret_type_str}')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'theta_ts.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=["theta_hist", "theta_ts"], sec_name=par.name, sub_dir=par.name, overall_label="theta",
                                     main_caption=fr"The figures above show the distribution of the predicted $\theta$. Panel (a) shows the histogram of all $\theta$ across time and firms. "
                                                  fr"Panel (b) shows the mean, median and quartiles across time. "
                                                  fr"We smooth the time series with a 12 month rolling average. ")

        ##################
        # make quantile plots
        ##################


        def func(x):
            return pd.qcut(x, 5, labels=False, duplicates='drop').values
        def av_geom(x):
            return (np.prod(1+x)**(1/x.shape[0])) -1


        df['port']=df.groupby('date')['pred'].transform(func)
        t=df.groupby(['port','date'])['ret'].mean().reset_index()
        p=t.groupby('port')['ret'].apply(av_geom)
        p = (1+p)**12 -1

        df['port']=df.groupby('date')['glb3_D30'].transform(func)
        t=df.groupby(['port','date'])['ret'].mean().reset_index()
        # b=t.groupby('port')['ret'].mean()
        b=t.groupby('port')['ret'].apply(av_geom)
        b = (1+b)**12  -1


        b.name = r'$Vilknoy$'
        p.name = r'$\theta_{nnet}$'
        q_final = pd.DataFrame([b, p]).T
        q_final.index += 1
        plt.plot(q_final.index, q_final.iloc[:, 0], label=q_final.columns[0], color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(q_final.index, q_final.iloc[:, 1], label=q_final.columns[1], color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(rf'{ret_type_str}')
        plt.xlabel('Quantile')
        plt.tight_layout()
        plt.savefig(self.dir_figs + 'quantile_portfolio.png')
        self.plt_show()

        Q_LIST = np.arange(0,1.1,0.1)
        df['q_pred'] = pd.qcut(df['pred'], Q_LIST, labels=False, duplicates='drop')
        t = df.groupby('q_pred')['ret'].mean()
        t_q = df.groupby('q_pred')['pred'].mean()
        plt.scatter(t.values, t_q.values, label='NNET', color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0], alpha=0.5)
        for i in range(len(t)):
            plt.annotate(list(t_q.index)[i], (t.values[i], t_q.values[i]))

        df['q_pred'] = pd.qcut(df['bench'], Q_LIST, labels=False, duplicates='drop')
        t = df.groupby('q_pred')['ret'].mean()
        t_q = df.groupby('q_pred')['bench'].mean()
        plt.scatter(t.values, t_q.values, label=r'$\theta=1.0$', color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1], alpha=0.5)
        for i in range(len(t)):
            plt.annotate(list(t_q.index)[i], (t.values[i], t_q.values[i]))

        y = np.arange(-0.05, 0.05, 0.001)
        plt.plot(y, y)
        plt.grid(True)
        plt.legend()
        plt.xlabel('Average realized returns')
        plt.ylabel(r'Average predicted returns')
        plt.tight_layout()

        plt.savefig(self.dir_figs + 'quantile_v2.png')

        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=["quantile_portfolio", "quantile_v2"], sec_name=par.name, sub_dir=par.name,
                                     main_caption=fr"The figures above show two construction of quantile's protfolios. "
                                                  fr"The left panel (a), shows the return of quantile portfolio for both definition of $\theta$. "
                                                  fr"On each day, we split the cross-section of firm based on the forecasted average {ret_type_str} and construct protfolio. "
                                                  fr"We take the average return of each quantile on each day to get daily returns per quantiles. "
                                                  fr"Finally, we take the average across time to get the quantile average daily log-return for each quantile."
                                                  fr"The right panel (b) sort the predictions into quantile panel wise and show on the x-axis the average realized return, an the y-axis "
                                                  fr"the average predicted return in each quantile.")

        # plt.scatter(df['pred'],df['ret'], color=didi.DidiPlot.COLOR[0], marker='+')
        # plt.show()
        #
        # plt.scatter(df['bench'],df['ret'], color=didi.DidiPlot.COLOR[0], marker='+')
        # plt.show()

        # ##################
        # # seaborn
        # ##################
        #
        # with sns.axes_style('white'):
        #     temp = df.copy()
        #
        #     temp=temp.loc[temp['ret'].abs()<=0.5,:]
        #     temp=temp.loc[temp['pred'].abs()<=0.1,:]
        #     r_ret = (temp['ret'].max()-temp['ret'].min())
        #     r_pred = (temp['pred'].max()-temp['pred'].min())
        #     temp['pred'] = (temp['pred']*r_ret)/r_pred
        #     temp[['pred','ret']].std()
        #
        #     sns.jointplot("ret", "pred", temp, kind='hex')
        #     plt.show()
        #
        #     sns.jointplot("ret", "bench", temp, kind='hex')
        #     plt.show()
