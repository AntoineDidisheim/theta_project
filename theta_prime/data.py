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
            TS_COL = [x for x in kelly.columns if 'M_' in x]
            CHAR_COL = [x for x in kelly.columns[4:] if 'M_' not in x]
            print('kelly',kelly['date'].min(),kelly['date'].max())
            print('ret',ret['date'].min(),ret['date'].max())

            # transform the label and merge with tr
            tr = self.load_tr_kelly()
            ind=tr[['date','gvkey']].duplicated()
            tr=tr.loc[~ind,:]
            kelly=kelly.merge(tr,how='left')
            ind=~kelly[['permno','date']].duplicated()
            kelly = kelly.loc[ind,:]
            ret=ret.dropna()
            ret['ym'] = ret['date'].dt.year*100+ret['date'].dt.month
            kelly['ym'] = kelly['date'].dt.year*100+kelly['date'].dt.month
            del kelly['date']
            kelly=ret.merge(kelly,how='left',on=['ym','permno'])
            del kelly['ym ']





            kelly.head()
            label = kelly.loc[:,['permno','date','ticker','ret1m']]
            ts = kelly.loc[:,TS_COL]
            indu = pd.get_dummies(kelly['sic2'])
            indu.columns = ['ind_'+str(x) for x in indu.columns]
            char = kelly.loc[:,CHAR_COL]



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

    def load_pred_feature(self, reload = False):
        target_days = self.par.data.H

        if reload:
            print('####################', 'start pred feature pr-processing')
            for v in ['mean', 'median']:
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
                t = df.groupby('permno')[f'{name_ret}_old'].rolling(252).aggregate([v]).reset_index().rename(columns={v: 'pred'})
                print(t)
                df = df.reset_index(drop=True)
                df = df.merge(t, how='left')

                ind = (~pd.isna(df[name_ret])) & (~pd.isna(df['pred']))
                df = df.loc[ind, :]
                df = df.reset_index(drop=True)

                TT = [20, 180, 252]
                Q = [0.25,0.75]
                pred_col = []
                df[f'err_{v}'] = (df['pred'] - df[f'{name_ret}_old']) ** 2
                pred_col.append('pred')
                for T in TT:
                    print(T)
                    df.index = df['date']
                    t = df.groupby('permno')[f'err_{v}'].rolling(T).agg(['mean', 'std']).reset_index()
                    t[f'err_{v}_mean_{T}'] = t.groupby('permno')['mean'].shift(1)
                    t[f'err_{v}_std_{T}'] = t.groupby('permno')['std'].shift(1)
                    pred_col.append(f'err_{v}_mean_{T}')
                    pred_col.append(f'err_{v}_std_{T}')
                    t = t.dropna()
                    del t['mean'], t['std']
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
            v_1 = 'mean'
            v_2 = 'median'
            df=pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_1}_H{target_days}.p').merge(pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_2}_H{target_days}.p'))
            # df.to_pickle(self.par.data.dir + f'raw_merge/price_feature.p')

        else:
            # df = pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature.p')
            print('start merge')
            v_1 = 'mean'
            v_2 = 'median'
            df=pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_1}_H{target_days}.p').merge(pd.read_pickle(self.par.data.dir + f'raw_merge/price_feature_{v_2}_H{target_days}.p'))
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

        for c in self.x_df.columns:
            if self.x_df[c].dtypes!='uint8':
                print('norm',c)
                self.x_df.loc[:, c] = (self.x_df[c] - self.x_df[c].mean()) / (self.x_df[c].max()-self.x_df[c].min())
            else:
                print('dont norm',c)
            # self.x_df.loc[:, c] = (self.x_df[c] - self.x_df[c].mean()) / self.x_df[c].std()


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