import csv
import os
import time

import pandas as pd
from parameters import *
import didipack as didi
from matplotlib import pyplot as plt
import socket
import pyreadr

class Cleaner:
    def __init__(self, par = Params()):
        self.par = par

    def clean_crsp_from_ret(self, df, h=None):
        """
        CRSP gives returns "RET" in the following form: buy t-1, sell t.
        We transform these returns so that you buy at time t and sell at time t+h to conform with the rest of this package
        :param df: a dataframe at daily frequency with columns (minimum) PERMNO; RET; DATE;
        :param h: the frequency
        """
        if h is None:
            h = self.par.data.freq

        df.columns = [x.lower() for x in df.columns]
        assert 'permno' in df.columns, "CRSP needs the PERMNO columns to identify stocks, please download data which contains a columns PERMNO"
        assert 'ret' in df.columns, "this function needs the 'ret' columns from CRSP."
        assert 'date' in df.columns, "this function needs the 'date' columns from CRSP."

        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
        df['ret'] = df['ret'].fillna(0.0)
        df = df.sort_values(['permno', 'date']).reset_index(drop=True)
        ind = df[['permno', 'date']].duplicated(keep='first')
        df = df.loc[~ind, :]

        df['log_ret'] = np.log(df['ret'] + 1)
        # with the shift means buying today selling in t-days
        df[self.par.data.ret_col] = df.groupby('permno')['log_ret'].rolling(h).sum().shift(-h).reset_index()['log_ret']
        df[self.par.data.ret_col] = np.exp(df[self.par.data.ret_col]) - 1
        t = df.groupby('permno')['date'].shift(-h).reset_index()['date']
        tt = (t - df['date']).dt.days
        # remove days when we have a large number of missing days between two trading dates
        df.loc[tt > tt.quantile(0.99), self.par.data.ret_col] = np.nan
        del df['log_ret']
        return df





self = Cleaner(Params())












