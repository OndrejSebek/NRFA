from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import xgboost as xgb
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt

import matplotlib.pyplot as plt

import os
import requests

import QC_utils as qc_u

"""
NRFA/EA/MO data:
    level1: merged raw data from each src as timeseries pd.DataFrame()
    level2: merged combined timeseries df without gaps, subset of inps
            + inp/exp timeseries df, timelagged, no gaps, no transforms
    level3: ->kernets

preprocessing:
    exports inp data scalers

models:
    defines & trains keras/xgb models
    exports fit stats
    plots

"""
class NRFA:
    def __init__(self, station_id):
        """
        Init NRFA instance and check if ID valid, set easting + northing.

        Parameters
        ----------
        station_id : int/string
            NRFA station ID

        """
        # class var declarations
        self.station_id = str(station_id)
        self.station_ids = []

        self.nearby_NRFA = []
        self.nearby_NHA = []
        self.nearby_EA = []
        self.nearby_MORAIN = []
        
        self.east = 0
        self.north = 0
        
        self.data_loaded = 0
        
        self.EA_RG_meta = None
        
        self.inp = []
        self.col_labels = []
        self.exp = []
        
        self.scaler_inp = None
        
        self.test_split = False

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        
        self.x_cal = []
        self.x_val = []
        self.y_cal = []
        self.y_val = []
        
        self.y_mean = None
                
        self.y_mod_cal = []
        self.y_mod_val = []
        self.y_mod_test = []
        
        self.kf = None
        self.kfold_indices_cal = []
        self.kfold_indices_val = []
        
        self.model = None
        self.history = []
        
        self.cb_es = None
        self.cb_rlr = None
        
        self.cal_dataset = None
        self.val_dataset = None
        
        self.RMSE_df = []
        self.NSE_df = []
        # self.epoch = 0
        
        self.xgb_reg = None        
        self.xgb_feature_imps = []
        self.t_x_cats = []
    
        # check if target station available (ID on API)
        self.root = 'https://nrfaapps.ceh.ac.uk/nrfa/ws/'
        web_service_stations = 'station-ids?format=json-object'
        
        x = requests.get(self.root+web_service_stations).json()
        self.station_ids = x['station-ids']
        
        if self.station_id in str(self.station_ids):
            print('\n\n', f'station id ok ({station_id})', '\n')
            
            #  set base location 
            web_service_stinfo = f'station-info?station={self.station_id}&format=json-object'
            fields = '&fields=all'
            y = requests.get(self.root+web_service_stinfo+fields).json()
            
            self.east = y['data'][0]['easting']
            self.north = y['data'][0]['northing']
            
        else:
            print('invalid station id')


    ''' _____________________________ META _______________________________ '''       
    
    def fetch_meta(self):
        """
        Fetch NRFA and EA api metadata (station IDs, easting, northing).

        """
        # NRFA meta:
        nrfa_mt = []
        for i in self.station_ids:
            web_service_stinfo = f'station-info?station={i}&format=json-object'
            fields = '&fields=easting,northing'
            y = requests.get(self.root+web_service_stinfo+fields).json()
            
            east = y['data'][0]['easting']
            north = y['data'][0]['northing']
            
            nrfa_mt.append([i, east, north])
        
        NRFA_meta = pd.DataFrame(nrfa_mt)
        NRFA_meta.columns = ['NRFA_ID', 'easting', 'northing']
        NRFA_meta.to_csv('meta/NRFA_meta.csv', index=False)    
        
        # EA API meta:
        root = 'https://environment.data.gov.uk/flood-monitoring/id/stations?parameter=rainfall'
        data = requests.get(root).json()
        
        api_ids = pd.DataFrame(columns=['id', 'easting', 'northing'])
        for i in data['items']:
            if all(['easting' in i, 'northing' in i]):
                api_ids = api_ids.append({'id': i['notation'], 
                                          'easting': i['easting'], 
                                          'northing': i['northing']},
                                         ignore_index=True)
       
        api_ids.to_csv('meta/EA_API_meta_raw.csv', index=False)
        api_ids['id'] = api_ids['id'].apply(self.format_EA_ids_helper)
        api_ids.to_csv('meta/EA_API_meta.csv', index=False)
     
    
    def set_ids_radius(self, thr_NRFA, thr_EA, thr_MO): 
        """
        Identify nearby NRFA station and EA RG IDs in the specified radius, 
        set self.nearby_xxx. 

        Parameters
        ----------
        thr_NRFA : int
            maximum distance for NRFA stations [km]
        thr_EA : int
            maximum distance for EA RGs [km].
        thr_MO : int
            maximum distance for MO RGs [km]

        """
        # conv to [m] ~ easting, northing
        thr_NRFA *= 1000
        thr_EA *= 1000
        thr_MO *= 1000
           
        # NRFA station ids within *thr m
        meta_NRFA = pd.read_csv('meta/NRFA_meta_gdfliveonly_format.csv')        
        self.nearby_NRFA = []

        if thr_NRFA != 'catch':
            print(f'identifying NRFA stations within {thr_NRFA/1000} km radius')
            for i in meta_NRFA.index:
                east = meta_NRFA['easting'].loc[i]
                north = meta_NRFA['northing'].loc[i]
    
                dist = np.sqrt( (east-self.east)*(east-self.east) 
                               + (north-self.north)*(north-self.north) )
                if dist < thr_NRFA:
                    self.nearby_NRFA.append(str(meta_NRFA['NRFA_ID'].loc[i]))
        else:
            print('identifying NRFA stations within the catchment')
            for i in self.station_ids:
                if (str(i)[:2] == self.station_id[:2]) and (len(str(i)) == len(self.station_id)):
                    self.nearby_NRFA.append(str(i))                       
            
        print(f'stations found: {self.nearby_NRFA}', '\n')
        print(f'identifying EA gauges within {thr_EA/1000} km radius')
        
        # EA rainfall gauges
        self.EA_RG_meta = pd.read_csv('meta/COSMOS_meta_updated.csv')
        self.EA_RG_meta = self.EA_RG_meta[['API_ID',
                               'NHA_ID',
                               'easting',
                               'northing']].dropna().drop_duplicates()
        
        self.nearby_NHA = []
        self.nearby_EA = []
        for i in self.EA_RG_meta.index:
            east = self.EA_RG_meta['easting'].loc[i]
            north = self.EA_RG_meta['northing'].loc[i]

            dist = np.sqrt( (east-self.east)*(east-self.east) +
                           (north-self.north)*(north-self.north) )
            if dist < thr_EA:
                self.nearby_NHA.append(self.EA_RG_meta['NHA_ID'].loc[i])
                self.nearby_EA.append(self.EA_RG_meta['API_ID'].loc[i])
            
        print(f'gauges found: {self.nearby_NHA}', '\n')
        #print(f'EA ids: {self.nearby_EA}', '\n')
        
        # MORAIN
        print(f'identifying MORAIN gauges within {thr_MO/1000} km radius')
        meta_MORAIN = pd.read_csv('meta/MORAIN_meta_upd.csv')
        
        self.nearby_MORAIN = []
        for i in meta_MORAIN.index:
            east = meta_MORAIN['EASTING'].loc[i]
            north = meta_MORAIN['NORTHING'].loc[i]
            dist = np.sqrt( (east-self.east)*(east-self.east) +
                           (north-self.north)*(north-self.north) )
            if dist < thr_MO:
                self.nearby_MORAIN.append(str(meta_MORAIN['ID'].loc[i]))
        print(f'gauges found: {self.nearby_MORAIN}', '\n')
    
    
    def set_ids_updwnstream(self, dist=0):
        """
        Set nearby_NRFA based on upstream/downstream meta file, up to 
        *dist [km] away (along the watercourse).

        Parameters
        ----------
        dist : int
            maximum distance for target station [km], 0 to include all.
            The default is 0.

        """
        # meta = pd.read_excel('stations_upstream_downstream/nrfa_nearest_sites.xlsx',
        #                      sheet_name='nrfa_nearest_sites')
        
        meta = pd.read_csv('meta/NRFA_meta_gdfliveonly_format_updwnstream.csv')
        
        sub_dt = meta[meta['station'] == int(self.station_id)]
        
        if dist > 0:
            sub_dt = sub_dt[sub_dt['distance']<1000*dist]
        
        self.nearby_NRFA = list(sub_dt['related_station'].unique().astype(str))
        if self.station_id not in self.nearby_NRFA:
            self.nearby_NRFA.extend([self.station_id])
    
    
    def update_ids_local(self, empty_NHA=0):
        """
        Update self.nearby_NRFA based on local files (to remove stations with
        no data).

        Parameters
        ----------
        empty_NHA : int, optional
            1 to set nearby_NHA to empty list. The default is 0.

        """
        NRFA_ids = []
        for file in os.listdir(f'data/nrfa_raw/{self.station_id}'):
            # print(file)
            NRFA_ids.append(file[5:10])
        
        self.nearby_NRFA = NRFA_ids
        
        if empty_NHA:
            self.nearby_NHA = []

    def set_ids_local(self):
         x = pd.read_csv(f"_model_inps/xgbsearch_{self.station_id}.csv",
                         dtype=str)
         
         q = x['var'].str[-2] == '-'
         
         ok = x[~q].copy()
         z = x[q].copy()
         z['var'] = z['var'].str[:-2] 
         
         x = pd.DataFrame(ok.append(z)['var'].unique(),
                          columns=['var'], dtype=int)
         
         nrfa = x[x['var'] < 100000]['var'].tolist() + [int(self.station_id)]
         mo = x[x['var'] > 100000]['var'].tolist()
         
         self.nearby_NRFA = list(np.unique(nrfa))
         self.nearby_MORAIN = mo
         
         print(self.station_id, nrfa, mo)
        
    ''' ____________________________ / META ______________________________ '''       

        
    ''' _____________________________ DATA _______________________________ '''
       
    def fetch_NRFA(self, src):  
        """
        Fetch idetified NRFA historical flows from the API and export
        to level1.
        
        """
        # fetch timeseries
        data = pd.DataFrame()
        for i in self.nearby_NRFA:
            web_service_tseries = 'time-series?format=json-object'            
            data_type = f'&data-type={src}&station={i}'          
            z = requests.get(self.root+web_service_tseries+data_type).json()
            
            cur_dt = pd.DataFrame({str(i): z['data-stream'][1::2]}, index=z['data-stream'][::2])
            data = pd.concat([data, cur_dt], axis=1, sort=False)
            
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        if not os.path.exists(f'data/level1/{self.station_id}'):
            os.mkdir(f'data/level1/{self.station_id}')
        data.to_csv(f'data/level1/{self.station_id}/{self.station_id}_NRFA.csv')
        
    def fetch_NRFA_gdfpluslive(self, preqc=False):     
        data = pd.DataFrame()
        for i in self.nearby_NRFA:
            web_service_tseries = 'time-series?format=json-object'            
            data_type_nrfa = f'&data-type=gdf&station={i}'          
            data_type_live = f'&data-type=gdf-live&station={i}'          
            
            z = requests.get(self.root+web_service_tseries+data_type_nrfa).json()
            z_l = requests.get(self.root+web_service_tseries+data_type_live).json()
            
            z_dt = pd.DataFrame({str(i): z['data-stream'][1::2]}, index=z['data-stream'][::2])
            z_l_dt = pd.DataFrame({str(i): z_l['data-stream'][1::2]}, index=z_l['data-stream'][::2])
            
            # z_l_dt.loc[z_dt.index[0]:z_dt.index[-1]] = z_dt.loc[z_dt.index[0]:z_dt.index[-1]]
            z_x = z_l_dt.loc[z_dt.index[-1]:]
            z_f = pd.concat([z_dt, z_x.iloc[1:]],
                            axis=0, sort=False)
            
            data = pd.concat([data, z_f], axis=1, sort=False)
            
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # replace with preqc values
        if preqc:
            if int(self.station_id) in self.nearby_NRFA:
                qc_corr = pd.read_csv('meta/_NRFA_qc/gdf-live-audit-counts-2020-02-17.csv',
                                      index_col=1)
                qc_corr = qc_corr[qc_corr['STATION'] == int(self.station_id)][['FLOW_VALUES']]
                qc_corr.index = pd.to_datetime(qc_corr.index,
                                               format='%Y-%m-%d %H:%M:%S').normalize()
                
                qc_corr['orig'] = qc_corr['FLOW_VALUES'].apply(qc_u.get_orig)
                
                qc_cors = qc_corr[qc_corr["orig"] != "nan"][["orig"]]
                data.loc[qc_cors.index, self.station_id] = qc_cors["orig"]

        if not os.path.exists(f'data/level1/{self.station_id}'):
            os.mkdir(f'data/level1/{self.station_id}')
        data.to_csv(f'data/level1/{self.station_id}/{self.station_id}_NRFA.csv')
        
    def fetch_agg_EA(self):
        """
        Fetch (local) identified EA gauge rainfall and aggregate to daily
        from 30min. Export to level1.
        
        """
        gf_df = pd.DataFrame()

        if len(self.nearby_NHA) != 0:
            # load gauges df
            for gauge in self.nearby_NHA:
                gf = pd.read_csv(f'../EA_data/EA_rainfall_30min_agg/{gauge}.csv',
                                 skiprows=5, header=None)
                gf.columns=['date', gauge, 'note']
                gf['date'] = pd.to_datetime(gf['date'], format='%Y%m%d%H%M%S')
                gf = gf[['date', gauge]]
                
                if gf_df.empty:
                    gf_df = gf
                else:
                    gf_df = pd.merge(gf_df, gf, on='date', how='outer')
            
            #AGG
            # extract dates
            dates = gf_df['date'].loc[::48]
            dates = pd.DatetimeIndex(dates).normalize()
            #print(dates)
            
            # aggregate to daily form 30min
            gf_df = gf_df.groupby(gf_df.index // 48).sum()
            #print(gf_df)
            
            # set date as index
            gf_df = gf_df.set_index(dates)
            #print(gf_df)
                
        # export
        if not os.path.exists(f'data/level1/{self.station_id}'):
            os.mkdir(f'data/level1/{self.station_id}')
    
        gf_df[gf_df < 0] = np.nan
        gf_df.to_csv(f'data/level1/{self.station_id}/{self.station_id}_EA.csv')
      
        
    def fetch_MO(self):
        """
        Fetch MORAIN rainfall data and export to level1.

        """
        gf_df = pd.DataFrame()

        if len(self.nearby_MORAIN) != 0:
            # load gauges df
            morain_ids = self.nearby_MORAIN.copy()
            for gauge in morain_ids:
                gf = pd.read_csv(f'data/morain_raw/data/{gauge}.csv',
                                 index_col=0)
                gf.columns = [str(gauge)]
                gf.index = pd.to_datetime(gf.index)

                if len(gf.index.unique()) == len(gf.index):
                    if gf_df.empty:
                        gf_df = gf
                    else:                        
                        gf_df = pd.merge(gf_df, gf,
                                         left_index=True, right_index=True,
                                         how='outer')
                else: 
                    self.nearby_MORAIN.remove(str(gauge))
                    print(f'{gauge} excluded for duplicate dates')
                    
        # export
        if not os.path.exists(f'data/level1/{self.station_id}'):
            os.mkdir(f'data/level1/{self.station_id}')
    
                
        # drop duplicate indices ~ some stations 2 values for the same day?
        # mby just exclude station
        # gf_df = gf_df.loc[~gf_df.index.duplicated(keep='last')]
        
        gf_df[gf_df < 0] = np.nan
        # print(gf_df.columns.values)
        gf_df.to_csv(f'data/level1/{self.station_id}/{self.station_id}_MO.csv')
 
    ''' ___________________________ / DATA _______________________________ '''       
        
    
    ''' ________________________ PREPROCESSING ___________________________ '''   
     
    def merge_inps(self, src='MO', ratio=.9): 
        """
        Merge NRFA station & EA gauge data into inps df (doesn't set inp/exp).
        Exports level2 data.

        Parameters
        ----------
        src : string, optional
            Rainfall data source ['EA', 'MO']. The default is 'MO'.
        ratio : float, optional
            Data completeness/overlap threshold. The default is .9. All 
            stations with < *ratio % of target NRFA station data points
            are excluded.

        """
        if src == 'EA':
            data_EA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_EA.csv',
                                  index_col=0, header=0)
            self.nearby_NHA = list(data_EA.columns)
        elif src == 'MO':
            data_EA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_MO.csv',
                                  index_col=0, header=0)
            self.nearby_MORAIN = list(data_EA.columns)
        else:
            data_EA = pd.DataFrame()
            self.nearby_MORAIN = []
            self.nearby_EA = []
            
        data_NRFA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_NRFA.csv',
                                index_col=0, header=0)
        self.nearby_NRFA = list(data_NRFA.columns)
        
        if not data_EA.empty: 
            merged = pd.merge(data_EA, data_NRFA,
                              left_index=True, right_index=True,
                              how='outer')
        else: 
            merged = data_NRFA
        
        # indices (start/end for self.station) ~ ratio ~ inp removal
        top_idx = merged[self.station_id].dropna().index[0]
        bot_idx = merged[self.station_id].dropna().index[-1]
        
        for i in merged:
            if merged.loc[top_idx:bot_idx, i].dropna().shape[0] < ratio*merged.loc[top_idx:bot_idx, self.station_id].dropna().shape[0]:
                # remove col from inp data
                merged = merged.drop(i, axis=1)
                
                # remove station from inp list
                if i in self.nearby_NRFA:
                    self.nearby_NRFA.remove(i)
                    print(f'{i} removed (NRFA)')
                elif (src == 'EA') and (i in self.nearby_NHA):
                    self.nearby_NHA.remove(i) 
                    print(f'{i} removed (NHA)')
                elif (src == 'MO') and (i in self.nearby_MORAIN):
                    self.nearby_MORAIN.remove(i)
                    print(f'{i} removed (MO)')
                else:
                    print(f'{i} removed (!? not in self.nearby_xxx !?)')

                        
        if not os.path.exists(f'data/level2/{self.station_id}'):
            os.mkdir(f'data/level2/{self.station_id}')
        
        # remove incomplete days
        #
        merged.dropna(axis=0, inplace=True)

        if merged.shape[0] == 0:
            self.empty_merged_df = True
            print(merged.columns)
            print(self.nearby_NRFA, self.nearby_MORAIN)
            print('\n !empty merged df')
        else:
            self.empty_merged_df = False
            print('\n', f'{merged.index[0]} - {merged.index[-1]}', '\n')
            print(f'merged inps: {merged.columns.values}')
                
        merged.to_csv(f'data/level2/{self.station_id}/{self.station_id}_merged.csv')
        print(merged)


    def timelag_inps(self, t_x, lag_opt, src='MO'):
        """
        Load data, timelag all inps up to t-(*t_x) and set inp/exp.
        Reads level2 merged data, exports level2 inp/exp data.

        Parameters
        ----------
        t_x : int
            max timelag value (t-x)
        lag_opt : string
            Which inputs to timelag ['station', 'EA', 'EA&station', 'NRFA', 'all'].
        src : string, optional
            Rainfall data source ['EA', 'MO']. The default is 'MO'.

        """
        data = pd.read_csv(f'data/level2/{self.station_id}/{self.station_id}_merged.csv',
                           index_col=0, header=0)
        skip_lagging = 0
        
        # inps to lag
        if lag_opt == 'station':
            inps_to_lag = self.station_id
        elif lag_opt == 'EA':
            if src == 'MO':
                inps_to_lag = self.nearby_MORAIN.copy()
            else:
                inps_to_lag = self.nearby_NHA.copy()
        elif lag_opt == 'EA&station':
            if src == 'MO':
                inps_to_lag = self.nearby_MORAIN.copy()
                inps_to_lag.append(self.station_id)
            else:
                inps_to_lag = self.nearby_NHA.copy()
                inps_to_lag.append(self.station_id)
        elif lag_opt == 'all':
            if src == 'MO':
                inps_to_lag = self.nearby_MORAIN.copy()
                inps_to_lag.extend(self.nearby_NRFA)
            elif src == 'EA':
                inps_to_lag = self.nearby_NHA.copy()
                inps_to_lag.extend(self.nearby_NRFA)
            else:
                inps_to_lag = self.nearby_NRFA.copy()
        elif lag_opt == 'NRFA':
            inps_to_lag = self.nearby_NRFA.copy()
        else:
            print('invalid lag_opt\n')
            return
        
        print(f'inps to lag: {inps_to_lag}')
        
        # if no stations/gauges for current lag_opt
        if (t_x == 0) or (len(inps_to_lag) == 0):
            skip_lagging = 1
            merged = data
        else:
            # check timesteps
            td = pd.to_datetime(data.index[1:].values) - pd.to_datetime(data.index[:-1].values)
            
        if not skip_lagging:            
            # if no gaps 
            if td[td.days > 1].empty:
                merged = pd.DataFrame()
                for i in range(t_x):
                    # timelagged [-1]
                    inp_lagged = data[inps_to_lag].iloc[:-(i+1)]
                    inp_lagged.index = data.index[(i+1):]
                    
                    if merged.empty:
                        merged = data
                    
                    # merge 
                    merged = pd.merge(merged, inp_lagged,
                                      left_index=True, right_index=True,
                                      suffixes=('', '-'+str(i+1)),
                                      how='inner',
                                      sort=False)
            
            # elif gaps found
            else:
                print(f'gaps found: {td[td.days>1].shape[0]}', '\n')
                
                # get gap indexes
                pdtd = pd.DataFrame(td, columns=['td'])
                pdtd_ = pdtd[pdtd['td'].dt.days > 1]
                
                # add last DataFrame index
                td_indexes = pdtd_.index.append(pd.Index([pdtd.index[-1]+1]))
                
                merged = pd.DataFrame()
                
                # divide into sub-periods without gaps, do t_x lagging
                # and append to merged
                top_index = 0
                for bot_index in td_indexes:
                    sub_merged = pd.DataFrame()
                    for i in range(t_x):
                        # sub current sub-period without gaps
                        sub_data = data[inps_to_lag].iloc[top_index:(bot_index+1)]
                        #print(sub_data)
                        
                        # if timelag t-x larger than current rows, skip
                        # -> merge then creates empty df and current period is skipped 
                        if i >= sub_data.shape[0]:
                            #print('skipping', i, sub_data.shape[0], top_index, bot_index)
                            break
                        
                        # get timelagged [-1] for current sub-period
                        inp_lagged = sub_data[:-(i+1)]
                        inp_lagged.index = sub_data.index[(i+1):]                    
                        #print(inp_lagged)
                        
                        if sub_merged.empty:
                            sub_merged = data.iloc[top_index:(bot_index+1)]
            
                        # merge with sub-period df
                        sub_merged = pd.merge(sub_merged, inp_lagged,
                                              left_index=True, right_index=True,
                                              suffixes=('', '-'+str(i+1)),
                                              how='inner',
                                              sort=False)
    
                    # update bottom index    
                    top_index = bot_index+1
                    #print(sub_merged)
                    
                    # append to merged df
                    if merged.empty:
                        merged = sub_merged
                    else:
                        merged = merged.append(sub_merged, sort=False)
        
        print(merged)
        
        # divide inp/exp
        self.exp = merged[self.station_id]
        self.inp = merged.drop(self.station_id, axis=1)    
        
        # set data_loaded flag
        self.data_loaded = 1
                
        # set col labels
        self.col_labels = self.inp.columns

        # export 
        self.inp.to_csv(f'data/level2/{self.station_id}/{self.station_id}_inp.csv')
        self.exp.to_csv(f'data/level2/{self.station_id}/{self.station_id}_exp.csv',
                        header=True)
        
        
    def update_local_inps(self):
        cols = pd.read_csv(f"_model_inps/xgbsearch_{self.station_id}.csv",
                           dtype=str)
        
        inp = pd.read_csv(f"data/level2/{self.station_id}/{self.station_id}_inp.csv",
                          index_col=0)
        
        inp[cols['var'].tolist()].to_csv(f"data/level2/{self.station_id}/{self.station_id}_inp.csv")
        self.data_loaded = 0
    

    def merge_timelag_inps_subset(self, n_t_xs, src='MO'):
        """
        Timelag *n_t_xs most important inps based on [XGB GAIN], overwrites 
        .._merged_inp & .._merged_out files. Reads level1 data, exports level2
        inp/exp data. 

        Parameters
        ----------
        n_t_xs : int
            nr. of inp features to subset
        src : string, optional
            Rainfall data source ['EA', 'MO']. The default is 'MO'.

        """
        if n_t_xs > self.xgb_feature_imps.shape[0]:
            n_t_xs = self.xgb_feature_imps.shape[0]
            
        t_xs = self.xgb_feature_imps['colname'][:n_t_xs]
        t_xs[(t_xs.index[-1]+1)] = self.station_id
        
        t_x_cats = []
        for i in t_xs:
            if i[-2] != '-':
                cur_t_x = 0
                cur_inp = i                 
            else:
                cur_t_x = int(i[-1])
                cur_inp = i[:-2]
            
            t_x_cats.append([cur_inp, cur_t_x])
            
        self.t_x_cats = pd.DataFrame(t_x_cats, columns=['inp', 't_x']).sort_values('t_x')
        print(self.t_x_cats)

        # merge only subsetted inps:
        #   load level1 data from *src
        #   -> ¬data
        if src == 'EA':
            data_EA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_EA.csv',
                                  index_col=0, header=0)
        elif src == 'MO':
            data_EA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_MO.csv',
                                  index_col=0, header=0)
        else:
            data_EA = pd.DataFrame()
            self.nearby_MORAIN = []
            self.nearby_EA = []
            
        data_NRFA = pd.read_csv(f'data/level1/{self.station_id}/{self.station_id}_NRFA.csv',
                                index_col=0, header=0)
        
        if not data_EA.empty: 
            data = pd.merge(data_NRFA, data_EA,
                            left_index=True, right_index=True,
                            how='outer')
        else: 
            data = data_NRFA
        
        # subset target inps
        #
        data = data[self.t_x_cats['inp'].unique()]
        data = data.sort_index().dropna()
        
        # GAPS:
        #
        td = pd.to_datetime(data.index[1:].values) - pd.to_datetime(data.index[:-1].values)
        print(f'{td[td.days>1].shape[0]} gaps found', '\n')
        
        # get gap indices
        pdtd = pd.DataFrame(td, columns=['td'])
        pdtd_ = pdtd[pdtd['td'].dt.days > 1]
        
        # add last DataFrame index
        td_indices = pdtd_.index.append(pd.Index([pdtd.index[-1]+1]))
        
        # MERGE
        #
        merged = pd.DataFrame()
        top_index = 0
        for bot_index in td_indices:
            sub_merged = pd.DataFrame()
            for i in self.t_x_cats['t_x'].unique():
                inps_to_lag = self.t_x_cats['inp'].loc[self.t_x_cats['t_x'] == i]
                # sub current sub-period without gaps
                sub_data = data[inps_to_lag].iloc[top_index:(bot_index+1)]
                #print(sub_data)
                
                # if timelag t-x larger than current rows, skip
                # -> merge then creates empty df and current period is skipped 
                if i >= sub_data.shape[0]:
                    #print('skipping', i, sub_data.shape[0], top_index, bot_index)
                    break
                
                # get timelagged [-1] for current sub-period
                if i == 0:
                    sub_merged = sub_data
                else:
                    inp_lagged = sub_data.iloc[:-i]
                    inp_lagged.index = sub_data.index[i:] 
                    inp_lagged.columns = inp_lagged.columns+'-'+str(i)
                    #print(inp_lagged)
                
                    if sub_merged.empty:
                        sub_merged = inp_lagged.iloc[top_index:(bot_index+1)]
        
                    # merge with sub-period df
                    sub_merged = pd.merge(sub_merged, inp_lagged,
                                          left_index=True, right_index=True,
                                          suffixes=('', '-'+str(i)),
                                          how='inner',
                                          sort=False)

            # update bottom index    
            top_index = bot_index+1
            #print(sub_merged)
            
            # append to merged df
            if merged.empty:
                merged = sub_merged
            else:
                merged = merged.append(sub_merged, sort=False)    
            
        merged = merged.dropna()
        
        # divide inp/exp
        self.exp = merged[self.station_id]
        self.inp = merged.drop(self.station_id, axis=1)    
        
        # set data_loaded flag
        self.data_loaded = 1
                
        # set col labels
        self.col_labels = self.inp.columns

        # export 
        self.inp.to_csv(f'data/level2/{self.station_id}/{self.station_id}_inp.csv')
        self.exp.to_csv(f'data/level2/{self.station_id}/{self.station_id}_exp.csv',
                        header=True)


    def scale_split_traintest(self, n_traintest_split=400, ratio_calval_split=.25):
        """
        Load data if not done while timelagging, standardise inputs, set 
        inp/exp and separate cal/val subsets. Exports scalers (opt).

        Parameters
        ----------
        n_traintest_split : int, optional
            train/test split (last *n_traintest_split data points -> test).
            The default is 400. Negative values split on shuffled df. 0 for
            no split (only cal/val).
        ratio_calval_split : float, optional
            cal/val split ratio (shuffled). The default is .25.

        """
        self.test_split = True if n_traintest_split != 0 else False

        # load data here
        if not self.data_loaded:    
            self.exp = pd.read_csv(f'data/level2/{self.station_id}/{self.station_id}_exp.csv',
                                   index_col=0, header=0,
                                   squeeze=True)
            self.inp = pd.read_csv(f'data/level2/{self.station_id}/{self.station_id}_inp.csv',
                                   index_col=0, header=0)   
            
        # standardise inps to ~ 0 mean, 1 std
        self.scaler_inp = preprocessing.StandardScaler()
        x = self.scaler_inp.fit_transform(self.inp)
        y = self.exp.copy()
    
        # train test splits
        if self.test_split:
            if n_traintest_split > 0:
                # split on last *n_traintest_split data points, non-shuffled
                self.x_train = x[:-n_traintest_split]
                self.y_train = y.iloc[:-n_traintest_split]
            
                self.x_test = x[-n_traintest_split:]
                self.y_test = y.iloc[-n_traintest_split:]
            else:
                # do shuffled train/test split (to validate NN ens against)
                ratio_traintest_split = (n_traintest_split*-1)/len(y)
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, 
                                                                                        test_size=ratio_traintest_split, 
                                                                                        random_state=None,
                                                                                        shuffle=True)
        else:
            # skip train/test split (do only cal/val)
            self.x_train = x
            self.y_train = y
            
            self.x_test = []
            self.y_test = []
        
        # cal/val splits
        self.x_cal, self.x_val, self.y_cal, self.y_val = train_test_split(self.x_train, self.y_train, 
                                                                          test_size=ratio_calval_split, 
                                                                          random_state=None,
                                                                          shuffle=True)
        self.y_mean = y.mean()
        
        
    def scale_split_kfolds(self, cur_fold, n_folds):
        """
        K fold crossval + standardisation.
        
        Load data if not done while timelagging, standardise inputs, set 
        inp/exp and separate cal/val subsets. Exports scalers (opt).

        Parameters
        ----------
        n_folds : int
            total nr. of folds
        cur_fold : int
            current fold

        """
        self.test_split = False

        if not self.data_loaded:
            data = pd.read_csv(f'data/level2/{self.station_id}/{self.station_id}_merged.csv',
                               index_col=0, header=0)
    
            self.exp = data[self.station_id]
            self.inp = data.drop(self.station_id, axis=1) 
            
        # standardise inps to ~ 0 mean, 1 std
        self.scaler_inp = preprocessing.StandardScaler()
        inp = self.scaler_inp.fit_transform(self.inp)
        exp = self.exp.copy()

        # set KFold indices if first epoch (first fold) 
        #
        if cur_fold == 0:
            self.kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
        
            self.kfold_indices_cal = []
            self.kfold_indices_val = []
            for train_index, test_index in self.kf.split(self.inp):
                print("cal:", len(train_index), "val:", len(test_index))
                self.kfold_indices_cal.append(train_index)
                self.kfold_indices_val.append(test_index)
        
        # set cal/val inp/exp
        self.x_cal = inp[self.kfold_indices_cal[cur_fold]]
        self.x_val = inp[self.kfold_indices_val[cur_fold]]
        self.y_cal = exp.iloc[self.kfold_indices_cal[cur_fold]]
        self.y_val = exp.iloc[self.kfold_indices_val[cur_fold]]
        
        self.y_mean = exp.mean()
    
    
            
    ''' ______________________ / PREPROCESSING ___________________________ '''   
        
    
    ''' ___________________________ KERAS ________________________________ '''

    def keras_model(self, lr=1e-4):
        """
        Define keras model topology and callbacks.

        Parameters
        ----------
        lr : float, optional
            NN learning rate. The default is 1e-4.

        """
        # create model
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(self.x_val.shape[1],)),
        #tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(.1),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(1, activation='linear')])
 
        # early stopping callback
        self.cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                      patience=16, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)
        
        # reduce LR callback
        self.cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                           patience=8, verbose=0, mode='auto',
                                                           min_delta=0.0001, cooldown=0,
                                                           min_lr=0.00001)
        
        # convert to keras dataset
        self.cal_dataset = tf.data.Dataset.from_tensor_slices((self.x_cal, self.y_cal))
        self.cal_dataset = self.cal_dataset.batch(32)
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.val_dataset = self.val_dataset.batch(32)
        
        # compile
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                           loss=['mse'],
                           metrics=['RootMeanSquaredError'])  


    def keras_fit(self, ep=10000):
        """
        Fit (train) model on inp/exp datasets and save RMSE score for the
        current run.

        Parameters
        ----------
        ep : int
            nr. of epochs for training. The default is 10000.

        """
        self.history = self.model.fit(self.cal_dataset, epochs=ep,
                                      validation_data=self.val_dataset, 
                                      callbacks=[self.cb_es, self.cb_rlr],
                                      verbose=0)
      
        # mods (batch_size ~ memory usage)  
        self.y_mod_cal = self.model.predict(self.x_cal, batch_size=32)[:, 0]
        self.y_mod_val = self.model.predict(self.x_val, batch_size=32)[:, 0]
        
        if self.test_split:
            self.y_mod_test = self.model.predict(self.x_test, batch_size=32)[:, 0]


    def save_model(self, out_id):
        """
        Export keras model with inp feature scaler. 

        Parameters
        ----------
        out_id : int, optional
            ID for exported model & inp scaler, -1 to disable scaler save.

        """
        # export scaler
        if out_id != -1:
            if not os.path.exists(f'_models/{self.station_id}'):
                os.mkdir(f'_models/{self.station_id}')
                
            joblib.dump(self.scaler_inp, f'_models/{self.station_id}/scaler{out_id}.pkl')
        
        # export model
        self.model.save(f'_models/{self.station_id}/mod{out_id}.h5')
        

    def keras_plots(self):
        """
        Export keras ts/sc plots.
        
        """
        if not os.path.exists(f'plots/{self.station_id}'):
            os.mkdir(f'plots/{self.station_id}')
        if not os.path.exists(f'plots/{self.station_id}/keras'):
            os.mkdir(f'plots/{self.station_id}/keras')

        # plot cal (big)
        plt.figure(figsize=(100, 10))
        plt.plot(self.y_cal.values, label='obs')
        plt.plot(self.y_mod_cal, label='mod')
        plt.legend()
        plt.savefig(f'plots/{self.station_id}/keras/cal.png', dpi=300)
        plt.close()

        # plot val (big)
        plt.figure(figsize=(100, 10))
        plt.plot(self.y_val.values, label='obs')
        plt.plot(self.y_mod_val, label='mod')
        plt.legend()
        plt.savefig(f'plots/{self.station_id}/keras/val.png', dpi=300)
        plt.close()
        
        # plot learning history
        plt.plot(self.history.history['loss'], label='cal')
        plt.plot(self.history.history['val_loss'], label='val')
        plt.legend()
        plt.savefig(f'plots/{self.station_id}/keras/history.png', dpi=300)
        plt.close()
        
    ''' _________________________ / KERAS ________________________________ '''


    ''' __________________________ XGBoost _______________________________ '''

    def xgb_model_fit(self):
        """
        Define XGBoost model and fit to inp/exp data, set RMSE df.

        """
        self.xgb_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, 
                                        colsample_bynode=1, colsample_bytree=1, gamma=0,
                                        importance_type='gain', learning_rate=0.1, max_delta_step=0,
                                        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                                        n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
                                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                        silent=None, subsample=1, verbosity=1)

        # fit
        x_cal = pd.DataFrame(self.x_cal, columns = self.col_labels)
        x_val = pd.DataFrame(self.x_val, columns = self.col_labels)
        x_test = pd.DataFrame(self.x_test, columns = self.col_labels)
        
        self.xgb_reg.fit(x_cal, self.y_cal)
        
        # get feature importances
        self.xgb_feature_imps = pd.DataFrame([self.col_labels, self.xgb_reg.feature_importances_]).T
        self.xgb_feature_imps.columns = ['colname', 'feature_importance']
        self.xgb_feature_imps = self.xgb_feature_imps.sort_values('feature_importance', ascending=False)
        self.xgb_feature_imps.index = range(self.xgb_feature_imps.shape[0])
        self.xgb_feature_imps.to_csv('_model_inps/'+self.station_id+'.csv',
                                     index=False)
        
        # mods
        self.y_mod_cal = self.xgb_reg.predict(x_cal)
        self.y_mod_val = self.xgb_reg.predict(x_val)
        
        if self.test_split:
            self.y_mod_test = self.xgb_reg.predict(x_test)
        
      
    def xgb_plots(self):
        """
        Export XGB ts/sc/tree plots.

        """
        # check for directories
        if not os.path.exists(f'plots/{self.station_id}'):
            os.mkdir(f'plots/{self.station_id}')
        if not os.path.exists(f'plots/{self.station_id}/xgb'):
            os.mkdir(f'plots/{self.station_id}/xgb')
                
        # timeseries plot cal/cal
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(80, 10), sharey=True, dpi=300)
        ax1.plot(self.y_mod_cal, label='mod')
        ax1.plot(self.y_cal.values, label='obs')
        ax1.legend()
        ax2.plot(self.y_mod_cal-self.y_cal.values, label='residuals')
        ax2.legend()
        plt.savefig(f'plots/{self.station_id}/xgb/ts_cal.png')
        plt.close()
        
        # scatter plot val/val
        plt.figure(figsize=(10, 10), dpi=300)
        plt.scatter(self.y_cal, self.y_mod_cal, marker='x')
        plt.xlim(0,  max(self.y_cal))
        plt.ylim(0, max(self.y_cal))
        plt.savefig(f'plots/{self.station_id}/xgb/sc_cal.png')
        plt.close()
        
        # timeseries plot val/val
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(80, 10), sharey=True, dpi=300)
        ax1.plot(self.y_mod_val, label='mod')
        ax1.plot(self.y_val.values, label='obs')
        ax1.legend()
        ax2.plot(self.y_mod_val-self.y_val.values, label='residuals')
        ax2.legend()
        plt.savefig(f'plots/{self.station_id}/xgb/ts_val.png')
        plt.close()
        
        # scatter plot val/val
        plt.figure(figsize=(10, 10), dpi=300)
        plt.scatter(self.y_val, self.y_mod_val, marker='x')
        plt.xlim(0,  max(self.y_val))
        plt.ylim(0, max(self.y_val))
        plt.savefig(f'plots/{self.station_id}/xgb/sc_val.png')
        plt.close()
        
        # tree feature importance 
         #plt.figure(figsize=(50, 50), dpi=300)
        ax = xgb.plot_importance(self.xgb_reg, ax=None, height=0.2, xlim=None,
                                 ylim=None, title='Feature importance',
                                 xlabel='F score', ylabel='Features',
                                 importance_type='gain',
                                 max_num_features=None, grid=True,
                                 show_values=True)
        
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='both', which='minor', labelsize=3)
        plt.savefig(f'plots/{self.station_id}/xgb/tree.png', dpi=300, max_num_features=10)
        plt.close()
    
    ''' ________________________ / XGBoost _______________________________ '''
  
    
    ''' _________________________ helpers ________________________________ ''' 

    @staticmethod
    def format_EA_ids_helper(x):
        """
        Helper function to format EA ids to capital letters and (atleast)
        6 digit numeric codes.

        """
        if x != x.upper():
            x = x.upper()
        if len(x) < 6 and x.isdigit():
           zrs = 6-len(x)
           for i in range(zrs):
               x = '0' + x
        return x


    def EA_ids_on_api(self):
        """
        Check for identified EA gauges on the real-time rainfall API.

        """
        api_ids = pd.read_csv('meta/EA_API_meta.csv')
        
        self.nearby_NHA = []
        for i in self.nearby_EA:
            if i in api_ids['id'].values:
                self.nearby_NHA.append(i)
        
        print(f'nearby EA gauges: {self.nearby_EA}', '\n')
        print(f'present on API (NHA ID): {self.nearby_NHA}', '\n')

    ''' ________________________ / helpers _______________________________ '''
    
    
    ''' ________________________ FIT METRICS _____________________________ '''
    
    def NSE(self):
        """
        Calculate NSE for cal and val (+test) periods
        (self.y_mod_cal, self.y_mod_val, self.y_mod_test).

        Returns
        -------
        pd.DataFrame()
            NSE df with 2/3 fit vals and n_total rows & cols

        """
        NSE_cal = 1 - ( ((self.y_cal-self.y_mod_cal)*(self.y_cal-self.y_mod_cal)).sum()
                       /((self.y_cal-self.y_cal.mean())*(self.y_cal-self.y_cal.mean())).sum()
                       )
        
        NSE_val = 1 - ( ((self.y_val-self.y_mod_val)*(self.y_val-self.y_mod_val)).sum()
                       /((self.y_val-self.y_val.mean())*(self.y_val-self.y_val.mean())).sum()
                       )
        
        if self.test_split:
            NSE_test = 1 - ( ((self.y_test-self.y_mod_test)*(self.y_test-self.y_mod_test)).sum()
                            /((self.y_test-self.y_test.mean())*(self.y_test-self.y_test.mean())).sum()
                            )
            
        if self.test_split:
            self.NSE_df = pd.DataFrame({'NSE_cal': NSE_cal,
                                        'NSE_val': NSE_val,
                                        'NSE_test': NSE_test,
                                        'rows': self.y_cal.shape[0]+self.y_val.shape[0]+self.y_test.shape[0],
                                        'cols': self.x_cal.shape[1]}, index=[0]) 
        else:
            self.NSE_df = pd.DataFrame({'NSE_cal': NSE_cal,
                                        'NSE_val': NSE_val,
                                        'rows': self.y_cal.shape[0]+self.y_val.shape[0],
                                        'cols': self.x_cal.shape[1]}, index=[0]) 
            
        return self.NSE_df
    
    
    def RMSE(self):
        """
        Calculate N/RMSE for cal and val (+test) periods 
        (self.y_mod_cal, self.y_mod_val, self.y_mod_test).

        Returns
        -------
        pd.DataFrame()
            N/RMSE df with 2/3 fit vals and n_total rows & cols

        """
        rmse_cal = sqrt(mean_squared_error(self.y_cal.values, self.y_mod_cal))        
        nrmse_cal = rmse_cal/self.y_mean 
        
        rmse_val = sqrt(mean_squared_error(self.y_val.values, self.y_mod_val))
        nrmse_val = rmse_val/self.y_mean 
        
        if self.test_split:
            rmse_test = sqrt(mean_squared_error(self.y_test.values, self.y_mod_test))
            nrmse_test = rmse_test/self.y_mean 
        
            self.RMSE_df = pd.DataFrame({'nRMSE_cal': nrmse_cal,
                                         'nRMSE_val': nrmse_val,
                                         'nRMSE_test': nrmse_test,
                                         'rows': self.y_cal.shape[0]+self.y_val.shape[0]+self.y_test.shape[0],
                                         'cols': self.x_cal.shape[1]}, index=[0])
        else:
            self.RMSE_df = pd.DataFrame({'nRMSE_cal': nrmse_cal,
                                         'nRMSE_val': nrmse_val,
                                         'rows': self.y_cal.shape[0]+self.y_val.shape[0],
                                         'cols': self.x_cal.shape[1]}, index=[0])  
        
        return self.RMSE_df

    ''' _______________________ / FIT METRICS ____________________________ '''
