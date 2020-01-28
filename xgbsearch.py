import pandas as pd
import numpy as np

import NRFA_v3 as nrfa

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

 
''' xgbsearch '''   

def xgbsearch(station_ids, range_opts, range_rad_m, inp_opts, xgb_sub_n, runs, lr=0.0001, ep=10000):

    pars = [['station', 'range_opt', 'range_dist', 'inp_opt', 'rows', 'cols',
            'nRMSE_cal_xgb', 'nRMSE_val_xgb', 'NSE_cal_xgb', 'NSE_val_xgb',
            'rows_sub', 'cols_sub']
            +runs*['nRMSE_cal_NN', 'nRMSE_val_NN', 'NSE_cal_NN', 'NSE_val_NN']]   
    imps = pd.DataFrame()
    
    
    for station_id in station_ids:
        # init NRFA instance and fetch (NRFA, EA nad MO) ids
        x = nrfa.NRFA(station_id)
        
        # ___________________________ level1 ____________________________
        
        # for inp_opt in inp_opts:
        for range_radius in np.arange(10000, range_rad_m, 10000):
            for range_opt in range_opts:
                x.set_ids_radius(range_radius, range_radius, range_radius)
            
                # up/dwn stream NRFA ids
                if range_opt == 'updwn':
                    x.set_ids_updwnstream(0)     
                
                # re-fetch for current set of inps
                x.fetch_NRFA()
                # x.fetch_agg_EA()
                x.fetch_MO()
                
                # ____________________ level2 ___________________________
                
                for inp_opt in inp_opts:
                    # merge current inps
                    # out: lvl2 raw
                    x.merge_inps(inp_opt, ratio=.95)
                            
                    # skip if empty merged df
                    if x.empty_merged_df:
                        print('skipping empty merged')
                        continue
                            
                    
                    # timelag merged inps
                    # out: lvl2 inp/exp
                    x.timelag_inps(inp_opt, 5, 'all')
                    
                    # skip if inp empty 
                    # (~when t-x=0 and only 1 inp(=exp))
                    if x.inp.empty:
                        print('skipping empty inp')
                        continue
                    
                    # __________________ ML _____________________________
                    
                    # fit xgb for feature imps subsetting    
                    #
                    x.set_scale_inps(-1)  
                    x.xgb_model_fit()
    
    
                    for xgb_sub in ([-1]+xgb_sub_n):
                        # merge to feature importance dataframe
                        #
                        if imps.empty:
                            imps = x.xgb_feature_imps.iloc[:xgb_sub].copy()
                        else:
                            imps = pd.merge(imps, x.xgb_feature_imps.iloc[:xgb_sub],
                                            # left_index=True, right_index=True,
                                            on='colname',
                                            how='outer')
                        # print(imps)
                            
                        # append current pars
                        #
                        if xgb_sub == -1:
                            xgb_NSE = x.NSE()
                            xgb_RMSE = list(x.RMSE[['nRMSE_cal', 'nRMSE_val']].iloc[0])
                            xgb_rowcol = [(len(x.x_cal)+len(x.x_val)), len(x.x_cal[0])]
                            
                        c_pars = ([station_id, range_opt, range_radius, inp_opt]
                                  + xgb_rowcol + xgb_RMSE + xgb_NSE)
    
                        # xgb inp subsetting
                        #
                        if xgb_sub > 0:
                            x.merge_timelag_inps_subset('MO', xgb_sub)     
                        
                        # append total available n_rows for current subset
                        #
                        c_pars.extend([x.inp.shape[0], x.inp.shape[1]])
    
                        
                        # keras models for each run
                        # (+ cal/val split and stand)
                        #
                        for run in range(runs):
                            # x.set_scale_inps(-1)  
                            x.set_scale_inps_kf(runs, run)
                            
                            x.keras_model(lr)
                            x.keras_fit(ep)
                            
                            # x.model.save('_models/'+str(station_id)+'/mod'+str(run)+'.h5')
                            
                            # x.xgb_model_fit()
                            
                            # append fit stats (nRMSE + NSE)
                            # 
                            c_pars.extend(list(x.RMSE[['nRMSE_cal', 'nRMSE_val']].iloc[0])+x.NSE())
                            print('\n', x.station_id, range_opt, range_radius, 
                                  inp_opt, xgb_sub, run, x.NSE(), '\n')
                            
                        # append current set of pars to pars DF
                        #
                        pars.append(c_pars)
                
    # OUT
    #
    imps.columns = ['var']+list(range(imps.shape[1]-1))
    pars = pd.DataFrame(pars).T
    pars.columns = ['var']+list(range(imps.shape[1]-1))
    # print(imps)
    # print(pars)
    
    # calc average fit from kfold
    #
    out = imps.append(pars, ignore_index=True)     
    
    # print(out.columns)
    # print(out)
    
    avg_metrics = pd.DataFrame()
    for metric in ['nRMSE_cal_NN', 'nRMSE_val_NN', 'NSE_cal_NN', 'NSE_val_NN']:
        avg_metrics = avg_metrics.append(out[out['var'] == metric].drop('var', axis=1).astype(float).mean(), ignore_index=True)
    
    avg_metrics = pd.concat([pd.DataFrame(['nRMSE_cal_NN_avg', 'nRMSE_val_NN_avg', 'NSE_cal_NN_avg', 'NSE_val_NN_avg']), avg_metrics],
                            axis=1, ignore_index=True)
    
    avg_metrics.columns = ['var']+list(range(pars.shape[1]-1))

    out = out.append(avg_metrics, ignore_index=True) 
    out.to_csv('GS/xgbsearch.csv')

''' ______________________________________________________________________ '''


# station_ids, range_opts, range_rad_m, xgb_sub_n, runs

xgbsearch(['23011', '28015', '28022'],
          ['radius', 'updwn'],
          50001,
          ['MO', 'NRFA_only'],
          [10, 20, 30],
          5,
          lr=0.0001,
          ep=10000)



# xgbsearch(['45001'],
#           ['radius', 'updwn'],
#           50001,
#           ['NRFA_only', 'MO'],
#           [10, 20, 30],
#           2,
#           lr=0.0001,
#           ep=10000)



'''
23011, 28015, 28022, 28044, 29002, 30002, 32008, 33013, 33066, 
            34010, 34012, 34018, 35003, 37008, 38003, 38029, 39001, 39026,
            39056, 39065, 39125, 40017, 45001, 46005, 46014, 47006, 47019,
            48001, 49006, 52010, 54017, 54057, 54110, 75017, 76017, 76021
            
            , 'updwn'
            
'''

