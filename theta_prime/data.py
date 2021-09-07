import pandas as pd
import numpy as np
from parameters import *
import didipack as didi
import os
class Data:
    def __init__(self, par = Params()):
        self.par = par


    def load_mw(self,reload=False):
        if reload:
            os.listdir(f'{self.par.data.dir}bench/')
            df = pd.read_csv(f'{self.par.data.dir}bench/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
            df['date']=pd.to_datetime(df['date'])
            df=df[['date','permno','mw30']].rename(columns={'mw30':'mw'})
            df['permno'] = df['permno'].astype(int)
            df['mw']/=12
            df.to_pickle(f'{self.par.data.dir}bench/mw_daily.p')
        else:
            df = pd.read_pickle(f'{self.par.data.dir}bench/mw_daily.p')

        return df
    def load_vilknoy(self,reload=False):
        if reload:
            them = pd.read_csv(f'{self.par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})
            them['date']=pd.to_datetime(them['date'])
            them=them[['date','permno','glb2_D30']].rename(columns={'glb2_D30':'vilk'})
            them['vilk']/=12
            them.to_pickle(f'{self.par.data.dir}bench/glb_daily.p')
        else:
            them = pd.read_pickle(f'{self.par.data.dir}bench/glb_daily.p')

        return them

    def get_vilknoy_permno(self):
        df = self.load_vilknoy()
        t = df[['permno']].drop_duplicates()
        t['permno'] = t['permno'].astype(int)
        t.to_csv('data/permno.txt', index=False, header=False)
        return t

    def load_all_price(self, reload=False):
        if reload:
            # Minimum columns: ['PERMNO','CUSIP','TICKER', 'date', 'PRC', 'RET', 'CFACPR']
            # df = pd.read_csv(self.par.data.dir + '/raw/crsp_all.csv',nrows=100000)
            df = pd.read_csv(self.par.data.dir + '/raw/crsp_all.csv')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df.columns = [x.lower() for x in df.columns]
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df = df.loc[~pd.isna(df['ret']), :]
            df = df.sort_values(['permno', 'date']).reset_index(drop=True)

            # df[['permno', 'date']].drop_duplicates().shape[0] - df.shape[0]
            ind = df[['permno', 'date']].duplicated(keep='first')
            df=df.loc[~ind,:]

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
                ret_col.append(f'ret{h_name}_old')
                H_col.append(f'H{h_name}')

                # with the shift means buying today selling in t-days
                df[f'ret{h_name}'] = df.groupby('permno')['log_ret'].rolling(h).sum().shift(-h).reset_index()['log_ret']
                df[f'ret{h_name}'] = np.exp(df[f'ret{h_name}']) - 1

                # without the shift, to get the features
                df[f'ret{h_name}_old'] = df.groupby('permno')['log_ret'].rolling(h).sum().shift(1).reset_index()['log_ret']
                df[f'ret{h_name}_old'] = np.exp(df[f'ret{h_name}_old']) - 1

                t = df.groupby('permno')['date'].shift(-h).reset_index()['date']
                tt = (t - df['date']).dt.days
                # remove random skip in days
                df.loc[tt > tt.quantile(0.99), f'ret{h_name}'] = np.nan
                df[f'H{h_name}'] = tt
            df.round(4)

            df = df[['permno', 'ticker', 'date','prc'] + ret_col].rename(columns={'shrout': 'shares_outstanding'})
            df.to_pickle(self.par.data.dir + f'raw_merge/price.p')
        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/price.p')
        return df

    def load_pred_feature(self, reload = False):
        if reload:
            for v in ['mean', 'median']:
                print('###############',v)
                df = self.load_all_price()
                # df = df.head(1000000)
                df = df.sort_values(['date', 'permno']).reset_index(drop=True)
                ## mean ret over last 3 month
                df.index = df['date']
                t = df.groupby('permno')['ret1m_old'].rolling(252).aggregate([v]).reset_index().rename(columns={v: 'pred'})
                print(t)
                df = df.reset_index(drop=True)
                df = df.merge(t, how='left')

                ind = (~pd.isna(df['ret1m'])) & (~pd.isna(df['pred']))
                df = df.loc[ind, :]
                df = df.reset_index(drop=True)

                TT = [20, 180, 252]
                Q = [0.25,0.75]
                pred_col = []
                df[f'err_{v}'] = (df['pred'] - df['ret1m']) ** 2
                pred_col.append('pred')
                for T in TT:
                    print(T)
                    df.index = df['date']
                    t = df.groupby('permno')[f'err_{v}'].rolling(T).agg(['mean', 'std']).reset_index()
                    t[f'err_{v}_mean_{T}'] = t.groupby('permno')['mean'].shift(10)
                    t[f'err_{v}_std_{T}'] = t.groupby('permno')['std'].shift(10)
                    pred_col.append(f'err_{v}_mean_{T}')
                    pred_col.append(f'err_{v}_std_{T}')
                    t = t.dropna()
                    del t['mean'], t['std']
                    df = df.reset_index(drop=True)
                    df = df.merge(t, how='left')

                    # for q in Q:
                    #     df.index = df['date']
                    #     t = df.groupby('permno')[f'err_{v}'].rolling(T).quantile(q).reset_index()
                    #     t[f'err_{v}_Quantile{q}_{T}'] = t.groupby('permno')[f'err_{v}'].shift(1)
                    #     pred_col.append(f'err_{v}_Quantile{q}_{T}')
                    #     t = t.dropna()
                    #     del t[f'err_{v}']
                    #     df = df.reset_index(drop=True)
                    #     df = df.merge(t, how='left')

                df = df[['permno','date','ticker','ret1m']+pred_col]
                df = df.dropna()
                df = df.rename(columns={'pred':v+'_pred'})
                print(df.head())
                df.to_pickle(self.par.data.dir + f'raw_merge/price_feature_{v}.p')
            print('####### start merge')
            del df
            v_1 = 'mean'
            v_2 = 'median'
            df=pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_1}.p').merge(pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_2}.p'))
            df.to_pickle(self.par.data.dir + f'raw_merge/price_feature.p')

        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature.p')
        return df

    def load_additional_crsp(self,reload=False):
        if reload:
            df = pd.read_csv(self.par.data.dir + '/raw/crsp_add.csv')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df.columns = [x.lower() for x in df.columns]
            df['mkt_cap'] = df['prc']*df['shrout']
            df=df[['date','permno','mkt_cap']]
            df.to_pickle(self.par.data.dir+'/raw_merge/crsp_add.p')
        else:
            df = pd.read_pickle(self.par.data.dir+'/raw_merge/crsp_add.p')
        return df

    def load_internally(self):
        df = self.load_pred_feature()
        if self.par.data.cs_sample==CSSAMPLE.VILK:
            v = self.load_vilknoy()
            df=df.merge(v[['permno','date']])
        if self.par.data.var_subset is not None:
            col = list(df.columns[:4])+self.par.data.var_subset
            df = df[col]

        df = df.dropna().reset_index(drop=True)
        self.label_df = df.iloc[:,:4]
        self.x_df = df.iloc[:,4:]

        for c in self.x_df.columns:
            self.x_df.loc[:, c] = (self.x_df[c] - self.x_df[c].mean()) / (self.x_df[c].max()-self.x_df[c].min())


    def set_year_test(self, year):
        test_id = np.array(list(self.label_df.index[self.label_df['date'].dt.year == year]))
        np.random.shuffle(test_id)
        train_id = np.array(list(self.label_df.index[self.label_df['date'].dt.year < year-1]))
        np.random.shuffle(train_id)

        self.test_label_df = self.label_df.iloc[test_id, :].reset_index(drop=True)
        self.test_x_df = self.x_df.iloc[test_id, :].reset_index(drop=True)

        self.train_label_df = self.label_df.iloc[train_id, :].reset_index(drop=True)
        self.train_x_df = self.x_df.iloc[train_id, :].reset_index(drop=True)

        print(f'Set training year {year}', flush=True)



self = Data(Params())
# self.load_all_price(True)
# self.load_pred_feature(True)
# self.load_vilknoy(True)
# self.load_mw(True)
# self.load_additional_crsp(True)