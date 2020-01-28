from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import xgboost as xgb
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn import preprocessing

import matplotlib.pyplot as plt

import os
import requests


# Trains another neural net on otputs of
# an ensemble of nets
#
# -> no improvement
#
class nnens:
    def __init__(self, station_id, n_nets):
        self.station_id = str(station_id)
        self.nets = []
        self.scalers = []
    
        self.net_rmses = pd.read_csv('RMSEs/keras_RMSE_'+self.station_id+'.csv', index_col=0)
        self.net_rmses['best_nets'] = np.sqrt(self.net_rmses['cal']*self.net_rmses['cal'] + self.net_rmses['val']*self.net_rmses['val'])
        self.net_rmses = self.net_rmses.sort_values('best_nets')
        
        self.net_rmses = self.net_rmses[:n_nets]
        
        self.best_nets = self.net_rmses.index


        for net in self.best_nets:
            self.nets.append(tf.keras.models.load_model('_models/'+self.station_id+'/mod'+str(net)+'.h5'))        
            self.scalers.append(joblib.load('_models/'+self.station_id+'/scaler'+str(net)+'.pkl'))
            
            print('net & scaler', net, 'loaded')


    def prep_inps(self):
        self.inp = pd.read_csv('data/inps/'+self.station_id+'/'+self.station_id+'_inp_merged.csv',
                               index_col=0, header=0)
        
        self.exp = pd.read_csv('data/inps/'+self.station_id+'/'+self.station_id+'_exp_merged.csv',
                               index_col=0, header=0)
        
        net_preds = []
        for i, net in enumerate(self.best_nets):
            inp = self.scalers[i].transform(self.inp)
            net_preds.append(self.nets[i].predict(inp))
            #.append(net.predict(self.inp.values, batch_size=32))
        
        self.net_preds = pd.DataFrame(np.concatenate(net_preds, axis=1))
        self.net_preds.index = self.inp.index
        self.net_preds.columns = self.best_nets
        
        inp = self.inp.values
        exp = self.exp.values
        
        self.x_cal, self.x_val, self.y_cal, self.y_val = train_test_split(inp, exp, 
                                                                  test_size=0.3, 
                                                                  random_state=None,
                                                                  shuffle=True)

     
        
    def keras_model(self, lr):
        # create model
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(self.x_val.shape[1],)),
        #tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(.1),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(1, activation='linear')])
 
        # early stopping callback
        self.cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                      patience=20, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)
        
        # reduce LR callback
        self.cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                           patience=10, verbose=0, mode='auto',
                                                           min_delta=0.0001, cooldown=0,
                                                           min_lr=0)
        
        # convert to keras dataset
        self.cal_dataset = tf.data.Dataset.from_tensor_slices((self.x_cal, self.y_cal))
        self.cal_dataset = self.cal_dataset.batch(32)
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.val_dataset = self.val_dataset.batch(32)
        
        # compile
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=['mse'],
              metrics=['RootMeanSquaredError'])  
        
        # RMSE df
        self.RMSE = pd.DataFrame(columns=['RMSE_cal', 'RMSE_val', 'epoch', 'rows', 'cols'])
        self.epoch = 0
   
    
    
    def keras_fit(self, ep):
        self.history = self.model.fit(self.cal_dataset, epochs=ep,
                                      validation_data=self.val_dataset, 
                                      callbacks=[self.cb_es, self.cb_rlr],
                                      verbose=2)
 
        # cal/cal period (batch_size ~ memory usage while prediction)
        self.y_mod_cal = self.model.predict(self.x_cal, batch_size=32)
        self.rmse_cal = sqrt(mean_squared_error(self.y_cal, self.y_mod_cal))        
        
        # val/val period 
        self.y_mod_val = self.model.predict(self.x_val, batch_size=32)
        self.rmse_val = sqrt(mean_squared_error(self.y_val, self.y_mod_val))
        
        # 
        self.epoch += ep
        self.RMSE = self.RMSE.append({'RMSE_cal': self.rmse_cal,
                                      'RMSE_val': self.rmse_val,
                                      'epoch': self.epoch,
                                      'rows': self.y_cal.shape[0]+self.y_val.shape[0], 
                                      'cols': self.x_cal.shape[1]}, ignore_index=True)





y = nnens(46014, 10)
y.prep_inps()

y.keras_model(.0001)
y.keras_fit(10000)






















