import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb

from scipy.stats import gaussian_kde

import os
import joblib
import matplotlib.pyplot as plt
from scipy.stats import t


class Kernets:
    def __init__(self, station, n_nets):
        self.station_id = str(station)
        self.nets = []
        self.scalers = []
        self.n_inps = []
        
        self.net_rmses = pd.read_csv('RMSEs/keras_RMSE_'+self.station_id+'.csv', index_col=0)
        self.net_rmses['best_nets'] = np.sqrt(self.net_rmses['cal']*self.net_rmses['cal'] + self.net_rmses['val']*self.net_rmses['val'])
        self.net_rmses = self.net_rmses.sort_values('best_nets')
        
        self.net_rmses = self.net_rmses[:n_nets]
        
        self.best_nets = self.net_rmses.index
        
        print('\n', self.station_id)
        for net in self.best_nets:
            self.nets.append(tf.keras.models.load_model('_models/'+self.station_id+'/mod'+str(net)+'.h5'))        
            self.scalers.append(joblib.load('_models/'+self.station_id+'/scaler'+str(net)+'.pkl'))
            self.n_inps.append(self.nets[0].get_weights()[0].shape[0])
            print('net & scaler', net, 'loaded')
            
            
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
            
        self.inps = pd.read_csv('data/level2/'+self.station_id+'/'+self.station_id+'_inp_merged.csv', index_col=0)
        self.obs = pd.read_csv('data/level2/'+self.station_id+'/'+self.station_id+'_exp_merged.csv', index_col=0)#.values)


    def get_pred(self, bounds, conf=0.95):
        ndim = self.inps.ndim
        if self.inps.shape[ndim-1] != self.n_inps[0]:
            print('invalid n inps')
            print(self.inps.shape[ndim-1], '/', self.n_inps[0])
            return
        
        if len([bounds]) > 1:
            self.inps = self.inps[bounds[0]:bounds[1]]
            self.obs = self.obs[bounds[0]:bounds[1]]
            
        # keras nets pred
        self.pred = []        
        for i, net in enumerate(self.nets):
            inps = self.scalers[i].transform(self.inps)
            self.pred.append(net.predict(inps))
        
        # xgb pred
        #scaler = joblib.load('_models/'+self.station_id+'/scaler99.pkl')
        #inps = scaler.transform(self.inps)
        #self.pred.append(self.xgb_reg.predict(inps).reshape(-1, 1))
        
        # comb
        self.pred = np.concatenate(self.pred, axis=1)
        
        ''' weights '''
        # rmses = pd.read_csv('RMSEs/keras_RMSE_'+self.station_id+'.csv')['val']
        weights = self.net_rmses['best_nets'].sum()/self.net_rmses['best_nets']
        
        self.m_w = np.average(self.pred, weights=weights, axis=1)
        self.std_w = np.sqrt(np.average((self.pred-self.m_w[:,None])**2, weights=weights, axis=1))
        ''' _______ '''
        
        # conf intervals
        n = len(self.pred)
        
        # use mean and std
        #self.m = np.mean(self.pred, axis=1)
        #self.std = np.std(self.pred, axis=1)
        
        # use weighted average and std
        self.m = self.m_w
        self.std = self.std_w
        
        # set conf intervals
        self.h = self.std * t.ppf((1+conf)/2, n-1)       
        self.low = self.m-self.h
        self.high = self.m+self.h
        
        # kde
        # self.kdes = []
        # for i in self.pred:
        #     self.kdes.append(gaussian_kde(i, bw_method=(.5/self.pred.std(ddof=1))))
    
    
    def save_pred(self):
        # check for out directory
        if not os.path.exists('data/level3/'+self.station_id):
            os.mkdir('data/level3/'+self.station_id)
        
        self.m = pd.DataFrame(self.m, index=self.obs.index, columns=['nn_m'])
        self.std = pd.DataFrame(self.std, index=self.obs.index, columns=['nn_std'])
        
        merged_out = pd.merge(self.obs, self.m,
                              left_index=True, right_index=True)
        merged_out = pd.merge(merged_out, self.std,
                              left_index=True, right_index=True)
        
        merged_out.to_csv('data/level3/'+self.station_id+'/'+self.station_id+'_merged.csv')
        pd.DataFrame(self.pred, index=self.obs.index).to_csv('data/level3/'+self.station_id+'/'+self.station_id+'_mods.csv')
        
        # pd.DataFrame(self.pred).to_csv('data/level3/'+self.station_id+'/nn/x_nns.csv')
        # pd.DataFrame(self.m).to_csv('data/level3/'+self.station_id+'/nn/x_m.csv')
        # pd.DataFrame(self.std).to_csv('data/level3/'+self.station_id+'/nn/x_std.csv')
        # pd.DataFrame(self.high).to_csv('data/level3/'+self.station_id+'/nn/x_h.csv')
        # pd.DataFrame(self.low).to_csv('data/level3/'+self.station_id+'/nn/x_l.csv')
   
    
    def get_orig_exp(self):
        self.exp_orig = pd.read_csv('data/level3/'+self.station_id+'/comp/'+self.station_id+'_orig.csv', index_col=0)
        self.exp_orig = pd.merge(self.obs, self.exp_orig, left_index=True, right_index=True, how='outer')
        self.exp_orig['orig'] = self.exp_orig['orig'].fillna(self.exp_orig[self.station_id])
        self.exp_orig = self.exp_orig.dropna()
        
        # binary qcd == preqc
        self.difs = (self.exp_orig['orig'] != self.exp_orig[self.station_id])
        
        # preqcd 
        self.exp_orig = self.exp_orig[['orig']]
    
    
    def find_outliers(self, n_std=5, d_abs=5):
        # flag vals *n_std stds away from mean and with *d_abs absolute distance (mby redundant)
        exp_v = np.concatenate(self.exp_orig.values)
        #fl = (abs(self.m - self.obs) > n_std*abs(self.std)) & (abs(self.m - self.obs) > d_abs)    # use qcd exp - obs
        fl = (abs(self.m - exp_v) > n_std*abs(self.std)) & (abs(self.m - exp_v) > d_abs)        # use preqc exp - exp_v
       
        # self.flagged = self.obs.copy()
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
        if not os.path.exists('plots/'+self.station_id+'/comp/'):
            os.mkdir('plots/'+self.station_id+'/comp/')
            
        if n_dt == 0:
            n_dt = len(self.low)
        
        # mods for each x/10 nets
        # plt.figure(figsize=(8, 5), dpi=600)
        # plt.plot(self.pred)
        # #plt.plot(self.obs.values)
        # plt.savefig('f.jpg')
        # plt.close()
        
        # 3 cons vals outside KDE
        plt.figure(figsize=(30, 6), dpi=600)
        #plt.plot(self.m)
        plt.fill_between(range(len(self.low))[-n_dt:], self.low[-n_dt:], self.high[-n_dt:])
        plt.plot(range(len(self.low))[-n_dt:], self.flags_tr[-n_dt:], marker='x', markersize=8, c='red', linestyle='')
        #plt.plot(range(len(self.low))[-n_dt:], self.difs.values[-n_dt:], marker='+', markersize=8, c='red', linestyle='')
        #plt.plot(self.obs.values)
        plt.savefig('plots/'+self.station_id+'/comp/flags_kde_3cons.jpg')
        plt.close()
        
        # single vals outside KDE
        plt.figure(figsize=(30, 6), dpi=600)
        #plt.plot(self.m)
        plt.fill_between(range(len(self.low))[-n_dt:], self.low[-n_dt:], self.high[-n_dt:])
        plt.plot(range(len(self.low))[-n_dt:], self.flagged[-n_dt:], marker='x', markersize=8, c='red', linestyle='')
        #plt.plot(range(len(self.low))[-n_dt:], self.difs.values[-n_dt:], marker='+', markersize=8, c='red', linestyle='')
        #plt.plot(self.obs.values)
        plt.savefig('plots/'+self.station_id+'/comp/flags_kde.jpg')
        plt.close()

        
        