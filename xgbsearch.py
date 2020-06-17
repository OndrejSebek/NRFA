import pandas as pd
import numpy as np

import NRFA_v3 as nrfa

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


''' __________________________ xgbsearch _________________________________ '''

def xgbsearch(station_ids, range_opts, range_rad_m, inp_opts, inp_thresh,
              max_timelag, xgb_sub_n, runs, lr=0.0001, ep=10000):
    """
    Gridsearch for best NN parameters ~ fit. 
    
    I:      determines inp features, creates dataset to model
    II:     fits XGB model - benchmark, feature importance (subsetting)
    III:    trains NNs on 5fold crossval divided current dataset, determines 
            fit (averaged across folds)

    Exports fit metrics to gridsearch_*station_id*.csv for each NRFA station.

    Parameters
    ----------
    station_ids : list [int/str]
        NRFA station IDs
    range_opts : list [str]
        input feature selection ['radius', 'updwn']
    range_rad_m : list [int]
        distance from target NRFA station
    inp_opts : list [str]
        input rainfall data src ['MO', 'NRFA_only']
    inp_thresh : float
        input data completeness threshold (to be included) ~ taget station
    max_timelag : int
        max t-x timelag
    xgb_sub_n : list [int]
        input feature subsetting 
    runs : int
        nr. of NNs trained for each par combination
    lr : float, optional
        NN learning rate. The default is 0.0001.
    ep : int, optional
        NN training epoch threshold. The default is 10000. (x early stopping)

    """
    for station_id in station_ids:
        # (re)set pars and imps
        pars = [['station', 'range_opt', 'range_dist', 'inp_opt', 'rows', 'cols',
                'nRMSE_cal_xgb', 'nRMSE_val_xgb', 'NSE_cal_xgb', 'NSE_val_xgb',
                'rows_sub', 'cols_sub']
                +runs*['nRMSE_cal_NN', 'nRMSE_val_NN', 'NSE_cal_NN', 'NSE_val_NN']]   
        imps = pd.DataFrame()

        # init NRFA instance and fetch (NRFA, EA nad MO) ids
        x = nrfa.NRFA(station_id)
        
        """ __________________________ level1 ____________________________ """
        
        # for inp_opt in inp_opts:
        for range_radius in np.arange(range_rad_m[0], range_rad_m[1], range_rad_m[2]):
            for range_opt in range_opts:
                x.set_ids_radius(range_radius, range_radius, range_radius)
            
                # up/dwn stream NRFA ids
                if range_opt == 'updwn':
                    x.set_ids_updwnstream(0)     
                
                # re-fetch for current set of inps  
                x.fetch_NRFA('gdf')
                x.fetch_MO()
                
                """ ____________________ level2 __________________________ """
                
                for inp_opt in inp_opts:
                    # merge current inps
                    # out: lvl2 raw
                    x.merge_inps(inp_opt, ratio=inp_thresh)
                            
                    # skip if empty merged df
                    if x.empty_merged_df:
                        print('skipping empty merged')
                        continue
                            
                    # timelag merged inps
                    # out: lvl2 inp/exp
                    x.timelag_inps(max_timelag, 'all', inp_opt)
                    
                    # skip if inp empty 
                    # (~when t-x=0 and only 1 inp(=exp))
                    if x.inp.empty:
                        print('skipping empty inp')
                        continue
                    
                    """ __________________ ML ____________________________ """
                    
                    # fit xgb for feature imps subsetting    
                    x.scale_split_traintest(n_traintest_split=0)
                    x.xgb_model_fit()
    
                    for xgb_sub in ([-1]+xgb_sub_n):
                        # merge to feature importance dataframe
                        if imps.empty:
                            imps = x.xgb_feature_imps.iloc[:xgb_sub].copy()
                        else:
                            imps = pd.merge(imps, x.xgb_feature_imps.iloc[:xgb_sub],
                                            on='colname',
                                            how='outer')
                            
                        # append current pars
                        #   set vars if first
                        if xgb_sub == -1:
                            xgb_RMSE = list(x.RMSE()[['nRMSE_cal', 'nRMSE_val']].iloc[0])
                            xgb_NSE = list(x.NSE()[['NSE_cal', 'NSE_val']].iloc[0])
                            xgb_rowcol = [(len(x.x_cal)+len(x.x_val)), len(x.x_cal[0])]
                            
                        c_pars = ([station_id, range_opt, range_radius, inp_opt]
                                  + xgb_rowcol + xgb_RMSE + xgb_NSE)
    
                        # xgb inp subsetting
                        if xgb_sub > 0:
                            x.merge_timelag_inps_subset(xgb_sub, 'MO')
                        
                        # append total available n_rows for current subset
                        c_pars.extend([x.inp.shape[0], x.inp.shape[1]])
    
                        # keras models for each run
                        #   (+ kfold cal/val split and stand)
                        for run in range(runs):
                            x.scale_split_kfolds(run, runs)
                            
                            x.keras_model(lr)
                            x.keras_fit(ep)
                            
                            # append fit stats (nRMSE + NSE)
                            c_pars.extend(list(x.RMSE()[['nRMSE_cal', 'nRMSE_val']].iloc[0])
                                          +list(x.NSE()[['NSE_cal', 'NSE_val']].iloc[0]))
                            
                            # console print pars
                            print('\n', x.station_id, range_opt, range_radius, 
                                  inp_opt, xgb_sub, run,
                                  list(x.NSE()[['NSE_cal', 'NSE_val']].iloc[0]),
                                  '\n')
                            
                        # append current set of pars to pars DF
                        pars.append(c_pars)
                
        # OUT
        #
        imps.columns = ['var']+list(range(imps.shape[1]-1))
        pars = pd.DataFrame(pars).T
        pars.columns = ['var']+list(range(imps.shape[1]-1))
        # print(imps)
        # print(pars)
        
        # calc average fit from kfold
        out = imps.append(pars, ignore_index=True)     
        
        # print(out.columns)
        # print(out)
        
        avg_metrics = pd.DataFrame()
        for metric in ['nRMSE_cal_NN', 'nRMSE_val_NN', 'NSE_cal_NN', 'NSE_val_NN']:
            avg_metrics = avg_metrics.append(out[out['var'] == metric]
                                             .drop('var', axis=1)
                                             .astype(float)
                                             .mean(),
                                             ignore_index=True)
        
        avg_metrics = pd.concat([pd.DataFrame(['nRMSE_cal_NN_avg', 'nRMSE_val_NN_avg',
                                               'NSE_cal_NN_avg', 'NSE_val_NN_avg']),
                                 avg_metrics],
                                axis=1, ignore_index=True)
        
        avg_metrics.columns = ['var']+list(range(pars.shape[1]-1))
    
        out = out.append(avg_metrics, ignore_index=True) 
        out.to_csv(f'GS/xgbsearch_{station_id}.csv')

''' ______________________________________________________________________ '''


# station_ids, range_opts, range_rad_m, xgb_sub_n, runs

# xgbsearch(['47019'],
#           ['radius', 'updwn'],
#           [30, 51, 20],
#           ['MO', 'NRFA_only'],
#           .9,
#           5,
#           [16, 32],
#           5,
#           lr=0.0001,
#           ep=1)


'''
            23011, 28015, 28022, 28044, 29002, 30002, 32008, 33013, 33066, 
            34010, 34012, 34018, 35003, 37008, 38003, 38029, 39001, 39026,
            39056, 39065, 39125, 40017, 45001, 46005, 46014, 47006, 47019,
            48001, 49006, 52010, 54017, 54057, 54110, 75017, 76017, 76021
            
'''
