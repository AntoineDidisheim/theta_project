import pandas as pd
import numpy as np
from parameters import *
from ml_model import *
from data import *
import os
from matplotlib import pyplot as plt
import didipack as didi

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
    def __init__(self, par:Params):
        self.par = par
        self.paper = didi.LatexPaper(dir_='tex_res')

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
        self.paper.create_paper('theta',sec=['Results'],author="Antoine Didisheim, Fabio Trojani, Simon Scheidegger", title="Theta Project")


    def cv_training(self):
        model = NetworkTheta(self.par)
        res = []
        for i in range(10):
            model.data.move_shuffle()
            r2, theta, p, mse=model.get_perf_oos()
            model.train()
            r2_new, theta_new, p_new,mse_new=model.get_perf_oos()
            p_bench, r2_bench, mse_bench =model.get_bench_perf()


            r = model.data.test_label_df.copy()
            r['pred_no_train'] = p.numpy()
            r['pred'] = p_new.numpy()
            r['bench'] = p_bench.numpy()
            r['theta'] = theta_new
            res.append(r)
            print('########### r2')
            print('old',r2,'new',r2_new,'bench',r2_bench)
            print('########### mse')
            print('old',mse,'new',mse_new,'bench',mse_bench)
            model.create_network()
        df = pd.concat(res)
        df.to_pickle(self.res_dir+'df.p')

    def create_report_sec(self):
        par = self.par
        self.paper.append_text_to_sec(sec_name="Results", text=r"\input{sec/" + par.name + ".tex}" + '\n' + '\n')
        self.paper.create_new_sec(sec_name=par.name)
        self.paper.append_text_to_sec(sec_name=par.name, text="\\clearpage \n \n")
        self.paper.append_text_to_sec(sec_name=par.name, text=r"\subsection{"+self.par.name+"}")

        L = "Layer: ("
        for l in self.par.model.layers:
            L+=str(l)+','
        L = L[:-1]+')'

        model_description = '\n'+"In this section we present the results where the neural network architecture is defined as, \n \n" \
        r"\begin{enumerate}"+'\n' \
        rf"\item Learning rate {self.par.model.learning_rate}" +'\n'\
        rf"\item {L}" +'\n'\
        rf"\item Learning rate {self.par.model.learning_rate}" +'\n'\
        rf"\item Batch size {self.par.model.batch_size}" +'\n'\
        rf"\item Dropout {self.par.model.dropout}" +'\n'\
        rf"\item Activation {self.par.model.activation}" +'\n'\
        r"\end{enumerate}"+'\n'\
        rf"We use the dataset {self.par.data.dtype.name}"+'\n'
        self.paper.append_text_to_sec(sec_name=par.name, text=model_description)

        df = pd.read_pickle(self.res_dir+'df.p')

        df['error_bench'] = (df['ret']-df['bench']).abs()
        df['error_pred'] = (df['ret']-df['pred']).abs()

        ##################
        # r2 plots
        ##################

        def r2(df_):
            r2_pred = 1-((df_['ret'] - df_['pred']) ** 2).sum() / ((df_['ret'] -0.0) ** 2).sum()
            r2_bench = 1-((df_['ret'] - df_['bench']) ** 2).sum() / ((df_['ret'] -0.0) ** 2).sum()
            return pd.Series({'NNET':r2_pred,r'$\theta=1.0$':r2_bench})

        caption ="In the figure above, we show the "
        r2_all = (r2(df)*100).round(2)

        df['year'] = df['opt_date'].dt.year
        yy=df.groupby('year').apply(lambda x:r2(x))
        plt.plot(yy.index, yy.iloc[:,0], label=yy.columns[0],color = didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(yy.index, yy.iloc[:,1], label=yy.columns[1],color = didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(r'$R^2 * 100$')
        plt.xlabel('Year')
        plt.savefig(self.dir_figs+'r2_ts.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names="r2_ts",sec_name=par.name, sub_dir=par.name ,
                                main_caption=fr"The figure above shows out of sample $R^2$ multiplied by 100 of the neural network and $\theta=1$ benchmark for year in the sample."
                                             fr"The overal $R^2$ multiplied by 100 is equal to {r2_all.iloc[0]} for the NNET, and {r2_all.iloc[1]} for the benchmark.")

        ##################
        # prediction mean and quartiles
        ##################


        t = df.groupby('opt_date')[['pred','bench']].mean().rolling(12).mean()
        k=0
        plt.plot(t.index,t['pred'],label='NNET, mean',color=didi.DidiPlot.COLOR[0],linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.plot(t.index,t['bench'],label=r'$\theta=1$, mean',color=didi.DidiPlot.COLOR[1],linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.legend()
        plt.grid()
        plt.xlabel('Date')
        plt.ylabel('Predicted log(R)')
        plt.savefig(self.dir_figs+'mean_pred.png')
        self.plt_show()


        k=-1
        for q in [0.25,0.5,0.75]:
            t = df.groupby('opt_date')[['pred','bench']].quantile(q).rolling(12).mean()
            k+=1
            plt.plot(t.index,t['pred'],label=rf'NNET, q={q}',color=didi.DidiPlot.COLOR[0],linestyle=didi.DidiPlot.LINE_STYLE[k])
            plt.plot(t.index,t['bench'],label=rf'$\theta=1$, q={q}',color=didi.DidiPlot.COLOR[1],linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.grid()
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Predicted log(R)')
        plt.savefig(self.dir_figs+'q_pred.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=['mean_pred','q_pred'],sec_name=par.name, sub_dir=par.name, overall_label="forecast",
                                main_caption=fr"The figure above show the forecast of the NNET and benchmark across time. "
                                             fr"Panel (a) shows the mean predicted log-return while panel (b) shows the cross-sectional median and quartiles. "
                                             fr"We smooth the time series with a 12 month moving average.")

        ##################
        # MSE acrosss time
        ##################
        df['mse_bench']=((df['ret']-df['bench'])**2)
        df['mse_net'] = ((df['ret']-df['pred'])**2)

        mse_bench = np.round(df['mse_bench'].mean(),5)
        mse_net = np.round(df['mse_net'].mean(),5)


        yy=df.groupby('opt_date')[['mse_net','mse_bench']].mean().rolling(12).mean()
        yy['dMse'] = (yy['mse_net']-yy['mse_bench'])/yy['mse_bench']
        yy.columns = ['NNET',r'$\theta=1.0$','dMse']
        plt.plot(yy.index, yy.iloc[:,0], label=yy.columns[0],color = didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(yy.index, yy.iloc[:,1], label=yy.columns[1],color = didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(r'MSE')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig(self.dir_figs+'mse_ts.png')
        self.plt_show()





        plt.plot(yy.index, yy.iloc[:,2], label=yy.columns[0],color = didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.grid(True)
        plt.ylabel(r'$(MSE_{net}-MSE_{\theta=1.0})/MSE_{\theta=1.0}$')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig(self.dir_figs+'dmse_ts.png')
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=["mse_ts","dmse_ts"],sec_name=par.name, sub_dir=par.name, overall_label="mse",
                                main_caption=fr"The figure above shows out of sample MSE of the neural network and $\theta=1$ benchmark for year in the sample. "
                                             r"Panel (a) shows the average MSE of both models while panel (b) shows the relative error expressed in percentage of the benchmark error ($(MSE_{net}-MSE_{\theta=1.0})/MSE_{\theta=1.0}$). "
                                             r"We smooth all time series with a 12 months moving average. "
                                             fr"The average MSE on the full sample is equal to {mse_net} for the NNET, and {mse_bench} for the benchmark.")

        ##################
        # theta distribution
        ##################

        plt.hist(df['theta'],bins=100,color=didi.DidiPlot.COLOR[0])
        plt.xlabel(r'$\theta$')
        plt.tight_layout()
        plt.savefig(self.dir_figs+'theta_hist.png')
        self.plt_show()



        k=0

        t = df.groupby('opt_date')[['theta']].mean().rolling(12).mean()
        plt.plot(t.index, t['theta'], label=rf'Average $\theta$', color=didi.DidiPlot.COLOR[k], linestyle=didi.DidiPlot.LINE_STYLE[k])

        for q in [0.25,0.5,0.75]:
            t = df.groupby('opt_date')[['theta']].quantile(q).rolling(12).median()
            k+=1
            plt.plot(t.index,t['theta'],label=rf'$\theta$, q={q}',color=didi.DidiPlot.COLOR[k],linestyle=didi.DidiPlot.LINE_STYLE[k])
        plt.grid()
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Predicted log(R)')
        plt.tight_layout()
        plt.savefig(self.dir_figs+'theta_ts.png')
        self.plt_show()


        self.paper.append_fig_to_sec(fig_names=["theta_hist","theta_ts"],sec_name=par.name, sub_dir=par.name, overall_label="theta",
                                main_caption=fr"The figures above show the distribution of the predicted $\theta$. Panel (a) shows the histogram of all $\theta$ across time and firms. "
                                             fr"Panel (b) shows the mean, median and quartiles across time. "
                                             fr"We smooth the time series with a 12 month rolling average. ")

        ##################
        # make quantile plots
        ##################
        Q_LIST = np.arange(0,1.1,0.1)
        q_pred=[]
        q_bench =[]
        for d in df['opt_date'].sort_values().unique():
            t=df.loc[df['opt_date']==d,:].copy()
            t['q']=pd.qcut(t['pred'],Q_LIST,labels=False,duplicates='drop')
            t=t.groupby('q')['ret'].mean()
            t.name = d
            q_pred.append(t)

            t=df.loc[df['opt_date']==d,:].copy()
            t['q']=pd.qcut(t['bench'],Q_LIST,labels=False,duplicates='drop')
            t=t.groupby('q')['ret'].mean()
            t.name = d
            q_bench.append(t)
        q_pred = pd.DataFrame(q_pred)
        q_bench = pd.DataFrame(q_bench)

        b = q_bench.mean()
        b.name = r'$\theta=1.0$'
        p =q_pred.mean()
        p.name = r'$\theta=\phi(X)$'
        q_final =pd.DataFrame([b,p]).T
        q_final.index +=1
        plt.plot(q_final.index, q_final.iloc[:,0], label=q_final.columns[0],color = didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0])
        plt.plot(q_final.index, q_final.iloc[:,1], label=q_final.columns[1],color = didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1])
        plt.grid(True)
        plt.legend()
        plt.ylabel(r'log(R)')
        plt.xlabel('Quantile')
        plt.tight_layout()
        plt.savefig(self.dir_figs+'quantile_portfolio.png')
        self.plt_show()




        df['q_pred'] = pd.qcut(df['pred'], Q_LIST, labels=False, duplicates='drop')
        t = df.groupby('q_pred')['ret'].mean()
        t_q =df.groupby('q_pred')['pred'].mean()
        plt.scatter(t.values, t_q.values, label='NNET', color=didi.DidiPlot.COLOR[0], linestyle=didi.DidiPlot.LINE_STYLE[0],alpha=0.5)
        for i in range(len(t)):
            plt.annotate(list(t_q.index)[i], (t.values[i],t_q.values[i]))

        df['q_pred'] = pd.qcut(df['bench'], Q_LIST, labels=False, duplicates='drop')
        t = df.groupby('q_pred')['ret'].mean()
        t_q =df.groupby('q_pred')['bench'].mean()
        plt.scatter(t.values, t_q.values, label=r'$\theta=1.0$', color=didi.DidiPlot.COLOR[1], linestyle=didi.DidiPlot.LINE_STYLE[1],alpha=0.5)
        for i in range(len(t)):
            plt.annotate(list(t_q.index)[i], (t.values[i],t_q.values[i]))

        y=np.arange(-0.1,0.1,0.001)
        plt.plot(y,y)
        plt.grid(True)
        plt.legend()
        plt.xlabel('Average realized returns')
        plt.ylabel(r'Average predicted returns')
        plt.tight_layout()

        plt.savefig(self.dir_figs + 'v_2.png')
        plt.show()
        self.plt_show()

        self.paper.append_fig_to_sec(fig_names=["quantile_portfolio","quantile_v2"], sec_name=par.name, sub_dir=par.name,
                                     main_caption=fr"The figures above show two construction of quantile's protfolios. "
                                                  fr"The left panel (a), shows the return of quantile portfolio for both definition of $\theta$. "
                                                  fr"On each day, we split the cross-section of firm based on the forecasted average log(R) and construct protfolio. "
                                                  fr"We take the average return of each quantile on each day to get daily returns per quantiles. "
                                                  fr"Finally, we take the average across time to get the quantile average daily log-return for each quantile."
                                                  fr"The right panel (b) sort the predictions into quantile panel wise and show on the x-axis the average realized return, an the y-axis "
                                                  fr"the average predicted return in each quantile.")

