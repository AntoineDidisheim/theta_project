import pandas as pd
import numpy as np
from parameters import *
import didipack as didi
import os
import sqlite3
from tqdm import tqdm
class Data:
    def __init__(self, par = Params()):
        self.par = par


    def load_mw(self,reload=True):
        if reload:
            df = pd.read_csv(f'{self.par.data.dir}bench/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
            df['date']=pd.to_datetime(df['date'])
            df['permno'] = df['permno'].astype(int)
            if self.par.data.H == 20:
                df = df[['date', 'permno', 'mw30']].rename(columns={'mw30': 'mw'})
                df['mw'] /= 12
            if self.par.data.H == 60:
                df = df[['date', 'permno', 'mw91']].rename(columns={'mw91': 'mw'})
                df['mw'] /= 4
            if self.par.data.H == 120:
                df = df[['date', 'permno', 'mw182']].rename(columns={'mw182': 'mw'})
                df['mw'] /= 2

            df.to_pickle(f'{self.par.data.dir}bench/mw_daily.p')
        else:
            df = pd.read_pickle(f'{self.par.data.dir}bench/mw_daily.p')
        return df




    def load_kelly(self):
        df = pd.read_pickle(f'{self.par.data.dir}/Data.pkl')
        return df

    def load_vilknoy(self,reload=True):
        if reload:
            them = pd.read_csv(f'{self.par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})
            them['date']=pd.to_datetime(them['date'])
            if self.par.data.H == 20:
                them=them[['date','permno','glb2_D30']].rename(columns={'glb2_D30':'vilk'})
                them['vilk']/=12
            if self.par.data.H == 60:
                them=them[['date','permno','glb2_D91']].rename(columns={'glb2_D91':'vilk'})
                them['vilk']/=4
            if self.par.data.H == 120:
                them=them[['date','permno','glb2_D182']].rename(columns={'glb2_D182':'vilk'})
                them['vilk']/=2
            them.to_pickle(f'{self.par.data.dir}bench/glb_daily_{self.par.data.H}.p')
        else:
            them = pd.read_pickle(f'{self.par.data.dir}bench/glb_daily_{self.par.data.H}.p')

        return them

    def get_vilknoy_permno(self):
        df = self.load_vilknoy()
        t = df[['permno']].drop_duplicates()
        t['permno'] = t['permno'].astype(int)
        t.to_csv('data/permno.txt', index=False, header=False)
        return t


    def load_translator_gvkey_permno_basic(self, reload=False):
        tr = pd.read_csv(self.par.data.dir+'permno_gvkey.csv')
        tr.columns = [x.lower() for x in tr.columns]
        tr=tr.loc[:,['gvkey','lpermno','linkdt','linkenddt']]
        tr['linkdt']=pd.to_datetime(tr['linkdt'],format='%Y%m%d')
        tr['linkenddt']=pd.to_datetime(tr['linkenddt'],format='%Y%m%d',errors='coerce')
        tr['linkenddt'] = tr['linkenddt'].fillna(pd.to_datetime(2100,format='%Y'))
        return tr

    def load_tr_kelly(self,reload=False):
        if reload:
            df = self.load_kelly()
            tr = self.load_translator_gvkey_permno_basic()

            # Make the db in memory
            conn = sqlite3.connect(':memory:')
            # write the tables
            df[['date', 'gvkey']].to_sql('df', conn, index=False)
            tr.to_sql('tr', conn, index=False)
            #write a query to merge between the date only
            qry = '''
                select  
                    *
                from
                    df join tr on
                    df.date between linkdt and linkenddt AND df.gvkey = tr.gvkey
                '''
            tr = pd.read_sql_query(qry, conn)

            # removing extra gvkey
            tr=tr[['gvkey','date','lpermno']].iloc[:,1:]
            tr['date']=pd.to_datetime(tr['date'])
            tr.columns = ['gvkey','date','permno']

            tr.to_pickle(self.par.data.dir + f'raw_merge/tr_kelly.p')
            print('finish',flush=True)
        else:
            tr = pd.read_pickle(self.par.data.dir + f'raw_merge/tr_kelly.p')
        return tr


    def load_all_price(self, reload=False):
        if reload:
            print('start pre-process price')
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
            H = [1, 20,60,120]
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


            df = df[['permno', 'ticker', 'date','prc'] + ret_col].rename(columns={'shrout': 'shares_outstanding'})
            df.to_pickle(self.par.data.dir + f'raw_merge/price.p')
            print('finish')
        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/price.p')
        return df

    def load_compustat(self, reload = False):
        if reload:
            df = pd.read_csv(self.par.data.dir + '/raw/compustat.csv')
            df.columns = [x.lower() for x in df.columns]
            df['Market Value']=df['prcc_c']*df['csho']
            df['ROA'] = df['ni']/df['at']
            df['Net Profit Margin'] = df['ni']/df['sale']
            df['Price to Earnings'] = df['ni']/df['Market Value']
            df['Price to Sales'] = df['sale']/df['Market Value']
            df['Book to Market'] = df['bkvlps']/df['Market Value']
            df=df[['lpermno','fyear','ROA','Net Profit Margin','Price to Earnings','Price to Sales','Book to Market','Market Value']].rename(columns={'fyear':'year','lpermno':'permno'})
            df.to_pickle(self.par.data.dir + f'raw_merge/compustat.p')
        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/compustat.p')
        return df

    def load_feature_kelly(self, reload=False):
        target_days = self.par.data.H

        if reload:
            print('####################', 'start pred feature pr-processing KELLY')
            ret = self.load_all_price()
            # df = df.head(1000000)
            ret = ret.sort_values(['date', 'permno']).reset_index(drop=True)
            if target_days == 20:
                name_ret = 'ret1m'
            if target_days == 60:
                name_ret = 'ret3m'
            if target_days == 120:
                name_ret = 'ret6m'

            ret =  ret[['permno', 'date', 'ticker', name_ret]]

            kelly = self.load_kelly()
            print('kelly',kelly['date'].min(),kelly['date'].max())
            print('ret',ret['date'].min(),ret['date'].max())

            # reseting the days of some month because it's not the same first in our version
            ret['ym'] = ret['date'].dt.year*100+ret['date'].dt.month
            t=ret.groupby('ym')['date'].transform('min')
            ind=t==ret['date']
            ret=ret.loc[ind,:]
            ret = ret.reset_index(drop=True)
            ret['date']=pd.to_datetime(ret['ym'],format='%Y%m')


            kelly.head()
            label = kelly.loc[:,['gvkey','date']]
            ts = kelly.loc[:,[x for x in kelly.columns if 'M_' in x]]
            indu = pd.get_dummies(kelly['sic2'])
            indu.columns = ['ind_'+str(x) for x in indu.columns]
            char = kelly.loc[:,[x for x in kelly.columns[4:] if 'M_' not in x]]

            # transform the label and merge with tr
            tr = self.load_tr_kelly()
            ind=tr[['date','gvkey']].duplicated()
            tr=tr.loc[~ind,:]
            label=label.merge(tr,how='left')
            label=label.merge(ret,how='left')

            ind = ~pd.isna(label['ret1m'])
            print('perc keep', ind.mean())

            label = label.loc[ind,:].reset_index(drop=True)
            label = label.loc[:,['permno', 'date', 'ticker', 'ret1m']]
            indu = indu.loc[ind,:].reset_index(drop=True)
            char = char.loc[ind,:].reset_index(drop=True)
            ts = ts.loc[ind,:].reset_index(drop=True)

            df =pd.concat([label,char,ts,indu],axis=1)
            # ## todo, perhaps add the interactions
            # for i in tqdm(range(ts.shape[1])):
            #     ts.iloc[]

            df.to_pickle(self.par.data.dir + f'raw_merge/feature_kelly_H{target_days}.p')

        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/feature_kelly_H{target_days}.p')
        return df

    def load_kelly_bench(self,reload=False):
        if reload:
            df = pd.read_csv(self.par.data.dir + '/bench/Forecast_ML.csv')
            df['date'] = pd.to_datetime(df['date'],format='%Y%m%d')
            df.to_pickle(self.par.data.dir + '/bench/Forecast_ML.p')
        else:
            df = pd.read_pickle(self.par.data.dir + '/bench/Forecast_ML.p')
        return df

    def load_lstm_feature(self, reload=False):
        target_days = self.par.data.H

        if reload:
            df = self.load_all_price()
            # df = df.head(1000000)
            pred_col = []

            print('####################', 'start pred feature pr-processing')
            for v in ['mean', 'median']:
                print('###############', v)
                if target_days == 20:
                    name_ret = 'ret1m'
                if target_days == 60:
                    name_ret = 'ret3m'
                if target_days == 120:
                    name_ret = 'ret6m'
                df = df.sort_values(['date', 'permno']).reset_index(drop=True)
                ## mean ret over last 3 month
                df.index = df['date']
                t = df.groupby('permno')[f'{name_ret}_old'].rolling(252).aggregate([v]).reset_index().rename(columns={v: f'pred_{v}'})
                print(t)
                df = df.reset_index(drop=True)
                df = df.merge(t, how='left')

                ind = (~pd.isna(df[name_ret])) & (~pd.isna(df[f'pred_{v}']))
                df = df.loc[ind, :]
                df = df.reset_index(drop=True)

                df[f'err_{v}'] = (df[f'pred_{v}'] - df[f'{name_ret}_old']) ** 2
                pred_col.append(f'pred_{v}')
                pred_col.append(f'err_{v}')

            df = df.sort_values(['permno','date']).reset_index(drop=True)
            # start by pushing everything into the future, once.
            for c in pred_col:
                pass
            df = df.dropna()

            print('start creating the extra shift')
            df[c]=df.groupby('permno')[c].shift(1)
            T = np.arange(20,20*13,20)
            m_name = {}
            for c in pred_col:
                l = [c]
                for t in T:
                    n = c+f'_{t}'
                    l.append(n)
                    df[n] = df.groupby('permno')[c].shift(t)
                m_name[c] = l
            df = df.dropna()

            print('create the list for matrix stack')
            X = []
            for c in pred_col:
                X.append(df[m_name[c]].values)
            print('run the stack')
            X=np.stack(X,2)
            df = df[['permno', 'date', 'ticker', 'ret1m']]
            print('start saving')
            df.to_pickle(self.par.data.dir + f'raw_merge/label_LSTM_H{target_days}.p')
            np.save(arr=X,file=self.par.data.dir + f'raw_merge/X_LSTM_H{target_days}.npy')
            print('all done')
        else:
            df = pd.read_pickle(self.par.data.dir + f'raw_merge/label_LSTM_H{target_days}.p')
            X= np.load(file=self.par.data.dir + f'raw_merge/X_LSTM_H{target_days}.npz')
        return df, X

    def load_pred_feature(self, reload = False):
        target_days = self.par.data.H

        if reload:
            print('####################', 'start pred feature pr-processing')
            # for v in ['mean', 'median','true_ret']:
            # for v in ['median','true_ret']:
            for v in ['true_ret']:
                print('###############',v)
                if target_days == 20:
                    name_ret = 'ret1m'
                if target_days == 60:
                    name_ret = 'ret3m'
                if target_days == 120:
                    name_ret = 'ret6m'

                df = self.load_all_price()
                # df = df.head(1000000)
                df = df.sort_values(['date', 'permno']).reset_index(drop=True)
                ## mean ret over last 3 month
                df.index = df['date']
                if v == 'true_ret':
                    df[f'err_{v}'] = df['ret1']
                else:
                    t = df.groupby('permno')[f'{name_ret}_old'].rolling(252).aggregate([v]).reset_index().rename(columns={v: 'pred'})
                    print(t)
                    df = df.reset_index(drop=True)
                    df = df.merge(t, how='left')

                    ind = (~pd.isna(df[name_ret])) & (~pd.isna(df['pred']))
                    df = df.loc[ind, :]
                    df = df.reset_index(drop=True)
                    df[f'err_{v}'] = (df['pred'] - df[f'{name_ret}_old']) ** 2

                TT = [20, 180, 252]
                Q = [0.25,0.75]
                pred_col = []
                if v != 'true_ret':
                    pred_col.append('pred')
                for T in TT:
                    print(T)
                    df.index = df['date']
                    t = df.groupby('permno')[f'err_{v}'].rolling(T).agg(['mean', 'std','median']).reset_index()
                    t[f'err_{v}_mean_{T}'] = t.groupby('permno')['mean'].shift(1)
                    t[f'err_{v}_median_{T}'] = t.groupby('permno')['median'].shift(1)
                    t[f'err_{v}_std_{T}'] = t.groupby('permno')['std'].shift(1)
                    pred_col.append(f'err_{v}_mean_{T}')
                    pred_col.append(f'err_{v}_median_{T}')
                    pred_col.append(f'err_{v}_std_{T}')
                    t = t.dropna()
                    del t['mean'], t['std'],t['median']
                    df = df.reset_index(drop=True)
                    df = df.merge(t, how='left')

                    for q in Q:
                        df.index = df['date']
                        t = df.groupby('permno')[f'err_{v}'].rolling(T).quantile(q).reset_index()
                        t[f'err_{v}_Quantile{q}_{T}'] = t.groupby('permno')[f'err_{v}'].shift(1)
                        pred_col.append(f'err_{v}_Quantile{q}_{T}')
                        t = t.dropna()
                        del t[f'err_{v}']
                        df = df.reset_index(drop=True)
                        df = df.merge(t, how='left')

                df = df[['permno','date','ticker',name_ret]+pred_col]
                df = df.dropna()
                df = df.rename(columns={'pred':v+'_pred'})
                print(df.head())
                df.to_pickle(self.par.data.dir + f'raw_merge/price_feature_{v}_H{target_days}.p')
            print('####### start merge')
            del df

        # df = pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature.p')
        v_1 = 'median'
        # v_2 = 'mean'
        v_2 = 'true_ret'
        df = pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_1}_H{target_days}.p').merge(pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_2}_H{target_days}.p'))
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
        try:
            if self.par.data.cs_sample in [CSSAMPLE.VILK, CSSAMPLE.FULL]:
                df = self.load_pred_feature()
            if self.par.data.cs_sample == CSSAMPLE.KELLY:
                df = self.load_feature_kelly(False)
        except:
            print('Feature not pre-processed, starting now with prices:')
            self.load_all_price(True)
            print('Start now with the features')
            if self.par.data.cs_sample in [CSSAMPLE.VILK, CSSAMPLE.FULL]:
                df = self.load_pred_feature(True)
            if self.par.data.cs_sample == CSSAMPLE.KELLY:
                df = self.load_feature_kelly(True)

            print('finish with vilk and mw')
            self.load_mw(True)
            self.load_vilknoy(True)


        if self.par.data.cs_sample==CSSAMPLE.VILK:
            v = self.load_vilknoy()
            df=df.merge(v[['permno','date']])
        if self.par.data.var_subset is not None:
            col = list(df.columns[:4])+self.par.data.var_subset
            df = df[col]

        df = df.dropna().reset_index(drop=True)
        self.label_df = df.iloc[:,:4]
        self.x_df = df.iloc[:,4:]

        if self.par.data.cs_sample in [CSSAMPLE.VILK, CSSAMPLE.FULL]:
            print('norm')
            for c in self.x_df.columns:
                self.x_df.loc[:, c] = (self.x_df[c] - self.x_df[c].mean()) / (self.x_df[c].max()-self.x_df[c].min())
            # self.x_df.loc[:, c] = (self.x_df[c] - self.x_df[c].mean()) / self.x_df[c].std()

        C = self.x_df.columns
        for c in C:
            v = 'Quantile0.75'
            if v in c:
                low = c.split(v)[0]+'Quantile0.25'+c.split(v)[1]
                new = c.split(v)[0]+'QuantileRange'+c.split(v)[1]
                self.x_df[new] = self.x_df[c]-self.x_df[low]
                del self.x_df[c]



        print('Data used')
        print(self.x_df.columns)


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



# par = Params()
# self = Data(par)
# self.load_pred_feature(True)

