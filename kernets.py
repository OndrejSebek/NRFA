import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb

from scipy.stats import gaussian_kde

import os
import joblib
import matplotlib.pyplot as plt
from scipy.stats import t

import QC_utils as qc_u


class Kernets:
    def __init__(self, station, n_nets):
        """
        Init kernets instance and subset *n_nets best (~RMSE fit) NNs.

        Parameters
        ----------
        station : int/string
            NRFA station ID
        n_nets : int
            nr. of nets to subset
            
        """
        self.station_id = str(station)
        self.nets = []
        self.scalers = []
        self.n_inps = []
        
        self.net_rmses = pd.read_csv(f'RMSEs/keras_RMSE_{self.station_id}.csv',
                                     index_col=0)
        self.net_rmses['best_nets'] = np.sqrt(self.net_rmses['cal']*self.net_rmses['cal']
                                              + self.net_rmses['val']*self.net_rmses['val'])
        self.net_rmses = self.net_rmses.sort_values('best_nets')
        
        self.net_rmses = self.net_rmses[:n_nets]
        
        self.best_nets = self.net_rmses.index
        
        print('\n', self.station_id)
        for net in self.best_nets:
            self.nets.append(tf.keras.models.load_model(f'_models/{self.station_id}/mod{net}.h5'))        
            self.scalers.append(joblib.load(f'_models/{self.station_id}/scaler{net}.pkl'))
            self.n_inps.append(self.nets[0].get_weights()[0].shape[0])
            print(f'net & scaler {net} loaded')
            
            
        # for file in os.listdir('_models/'+self.station_id):
        #     if file[0] != 's' and file[0] != 'b':
        #         if file[0] != 'x':           
        #             self.nets.append(tf.keras.models.load_model('_models/'+self.station_id+'/'+file))
        #             self.n_inps.append(self.nets[0].get_weights()[0].shape[0])
        #             print(file, 'loaded (net)')
        #         else:
        #             self.xgb_reg = xgb.XGBRegressor()
        #             self.xgb_reg.load_model('_models/'+self.station_id+'/'+file)
        #             print(file, 'loaded (xgb)')
            
        self.inp = pd.read_csv(f'data/level2{+self.station_id}/{self.station_id}_inp.csv',
                               index_col=0)
        self.exp = pd.read_csv(f'data/level2/{self.station_id}/{self.station_id}_exp.csv',
                               index_col=0)#.values)


    def get_mod(self, bounds=[], conf=0.95):
        """
        Model (self.m, self.std) with all subsetted (__init__) NNs 
        for given period.

        Parameters
        ----------
        bounds : [int, int]
            bounds used to subset data points
        conf : float, optional
            confidence level value, the default is 0.95
            
        """
        ndim = self.inp.ndim
        if self.inp.shape[ndim-1] != self.n_inps[0]:
            print('invalid n inps')
            print(f'{self.inp.shape[ndim-1]}/{self.n_inps[0]}')
            return
        
        if len([bounds]) > 1:
            self.inp = self.inp[bounds[0]:bounds[1]]
            self.exp = self.exp[bounds[0]:bounds[1]]
            
        # keras nets pred
        self.pred = []        
        for i, net in enumerate(self.nets):
            inps = self.scalers[i].transform(self.inp)
            self.pred.append(net.predict(inps))
        
        # xgb pred
        #scaler = joblib.load('_models/'+self.station_id+'/scaler99.pkl')
        #inps = scaler.transform(self.inp)
        #self.pred.append(self.xgb_reg.predict(inps).reshape(-1, 1))
        
        # comb
        self.pred = np.concatenate(self.pred, axis=1)
        
        # WEIGHTS
        #
        # rmses = pd.read_csv('RMSEs/keras_RMSE_'+self.station_id+'.csv')['val']
        weights = self.net_rmses['best_nets'].sum()/self.net_rmses['best_nets']
        
        self.m_w = np.average(self.pred, weights=weights, axis=1)
        self.std_w = np.sqrt(np.average((self.pred-self.m_w[:,None])**2, weights=weights, axis=1))
        
        # conf intervals
        # n = len(self.pred)
        
        # use mean and std
        #self.m = np.mean(self.pred, axis=1)
        #self.std = np.std(self.pred, axis=1)
        
        # use weighted average and std
        self.m = self.m_w
        self.std = self.std_w
        
        # set conf intervals
        # self.h = self.std * t.ppf((1+conf)/2, n-1)       
        # self.low = self.m-self.h
        # self.high = self.m+self.h
        
        # kde
        # self.kdes = []
        # for i in self.pred:
        #     self.kdes.append(gaussian_kde(i, bw_method=(.5/self.pred.std(ddof=1))))
    
    
    def save_mod_merged(self):
        """
        Export modelled level3 timeseries.
        
        """
        # check for out directory
        if not os.path.exists(f'data/level3/{self.station_id}'):
            os.mkdir(f'data/level3/{self.station_id}')
        
        qc = pd.read_csv(f'data/level3/{self.station_id}/{self.station_id}_qc.csv',
                         index_col=0)
        
        self.m = pd.DataFrame(self.m, index=self.exp.index, columns=['nn_m'])
        self.std = pd.DataFrame(self.std, index=self.exp.index, columns=['nn_std'])
        
        merged_out = pd.merge(qc, self.m,
                              left_index=True, right_index=True)
        merged_out = pd.merge(merged_out, self.std,
                              left_index=True, right_index=True)
        
        merged_out.to_csv(f'data/level3/{self.station_id}/{self.station_id}_merged.csv')
        pd.DataFrame(self.pred,
                     index=self.exp.index).to_csv(f'data/level3/{self.station_id}/{self.station_id}_mods.csv')
        
        # pd.DataFrame(self.pred).to_csv('data/level3/'+self.station_id+'/nn/x_nns.csv')
        # pd.DataFrame(self.m).to_csv('data/level3/'+self.station_id+'/nn/x_m.csv')
        # pd.DataFrame(self.std).to_csv('data/level3/'+self.station_id+'/nn/x_std.csv')
        # pd.DataFrame(self.high).to_csv('data/level3/'+self.station_id+'/nn/x_h.csv')
        # pd.DataFrame(self.low).to_csv('data/level3/'+self.station_id+'/nn/x_l.csv')
   
    
    def get_orig_exp(self):
        """
        REDUNDANT: Get (@self.exp_orig) original (preQC) timeseries.

        """
        self.exp_orig = pd.read_csv(f'data/level3/{self.station_id}/comp/{self.station_id}_orig.csv',
                                    index_col=0)
        self.exp_orig = pd.merge(self.exp, self.exp_orig,
                                 left_index=True, right_index=True,
                                 how='outer')
        self.exp_orig['orig'] = self.exp_orig['orig'].fillna(self.exp_orig[self.station_id])
        self.exp_orig = self.exp_orig.dropna()
        
        # binary qcd == preqc
        self.difs = (self.exp_orig['orig'] != self.exp_orig[self.station_id])
        
        # preqcd 
        self.exp_orig = self.exp_orig[['orig']]
    
    
    def find_outliers(self, n_std=5, d_abs=0):
        """
        REDUNDANT: Z-score + ABS outlier flagging.

        Parameters
        ----------
        n_std : float, optional
            Z-score (nr. of STDs) threshold. The default is 5.
        d_abs : float, optional
            ABS threshold. The default is 0.

        """
        # flag vals *n_std stds away from mean and with *d_abs absolute distance (mby redundant)
        exp_v = np.concatenate(self.exp_orig.values)
        #fl = (abs(self.m - self.exp) > n_std*abs(self.std)) & (abs(self.m - self.exp) > d_abs)    # use qcd exp - nn mod
        fl = ((abs(self.m-self.exp_orig.values) > n_std*abs(self.std).values) 
              & (abs(self.m.values-self.exp_orig.values) > d_abs))      # use preqc exp - nn mod
       
        # self.flagged = self.exp.copy()
        self.flagged = self.exp_orig.values
        self.flagged[~fl] = np.nan
        
        # kde
        # flag values outside estimated PDE
        self.flags = []
        for i, val in enumerate(exp_v):
            if self.kdes[i].evaluate(val) == 0:
                self.flags.append(val)
            else:
                self.flags.append(np.nan)
        
        # flag three consecutive values outside estimated PDE
        self.flags_tr = []
        for i, val in enumerate(self.flags):
            if i < len(self.flags)-1: 
                if all([val > 0, self.flags[i-1] > 0, self.flags[i+1] > 0]):
                    self.flags_tr.append(exp_v[i])
                else:
                    self.flags_tr.append(np.nan)
            else: 
                self.flags_tr.append(np.nan)
                
                
    def plots(self, n_dt):
        """
        REDUNDANT: Outlier flagging plots

        Parameters
        ----------
        n_dt : int
            subset last *n_dt data points (0 for no subsetting)

        """
        if not os.path.exists(f'plots/{self.station_id}/comp/'):
            os.mkdir(f'plots/{self.station_id}/comp/')
            
        if n_dt == 0:
            n_dt = len(self.low)
        
        # mods for each x/10 nets
        # plt.figure(figsize=(8, 5), dpi=600)
        # plt.plot(self.pred)
        # #plt.plot(self.exp.values)
        # plt.savefig('f.jpg')
        # plt.close()
        
        # 3 cons vals outside KDE
        plt.figure(figsize=(30, 6), dpi=600)
        #plt.plot(self.m)
        plt.fill_between(range(len(self.low))[-n_dt:], self.low[-n_dt:], self.high[-n_dt:])
        plt.plot(range(len(self.low))[-n_dt:], self.flags_tr[-n_dt:], marker='x', markersize=8, c='red', linestyle='')
        #plt.plot(range(len(self.low))[-n_dt:], self.difs.values[-n_dt:], marker='+', markersize=8, c='red', linestyle='')
        #plt.plot(self.exp.values)
        plt.savefig(f'plots/{self.station_id}/comp/flags_kde_3cons.jpg')
        plt.close()
        
        # single vals outside KDE
        plt.figure(figsize=(30, 6), dpi=600)
        #plt.plot(self.m)
        plt.fill_between(range(len(self.low))[-n_dt:], self.low[-n_dt:], self.high[-n_dt:])
        plt.plot(range(len(self.low))[-n_dt:], self.flagged[-n_dt:], marker='x', markersize=8, c='red', linestyle='')
        #plt.plot(range(len(self.low))[-n_dt:], self.difs.values[-n_dt:], marker='+', markersize=8, c='red', linestyle='')
        #plt.plot(self.exp.values)
        plt.savefig(f'plots/{self.station_id}/comp/flags_kde.jpg')
        plt.close()
