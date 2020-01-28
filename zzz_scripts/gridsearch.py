import pandas as pd
import numpy as np

import NRFA_v3 as nrfa

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

 
''' gridsearch '''   

def gridsearch(station_ids, range_opts, range_bounds, merge_opts, ratio_bounds, t_x_bounds, t_x_opts, runs, lr=0.0001, ep=10000):
    
    big_RMSE = []
    pars = []
    
    for station_id in station_ids:
        # init instance and fetch NRFA, EA nad MO ids
        x = nrfa.NRFA(station_id)
        
        for range_opt in range_opts:                    
            for range_radius in np.arange(range_bounds[0], range_bounds[1], range_bounds[2]):
                x.set_ids_radius(range_radius, range_radius, range_radius)
                   
                # re-fetch for current set of inps
                x.fetch_NRFA()
                # x.fetch_agg_EA()
                x.fetch_MO()
                
                # up/dwn stream
                if range_opt == 'updwn':
                    x.set_ids_updwnstream(0)
                
                for merge_opt in merge_opts:
                    for ratio in np.arange(ratio_bounds[0], ratio_bounds[1], ratio_bounds[2]):
                        # merge current inps
                        # out: lvl2 raw
                        x.merge_inps(merge_opt, ratio)
                        
                        # skip if empty merged df
                        if x.empty_merged_df:
                            print('skipping empty merged')
                            continue
                        
                        for t_x in range(t_x_bounds[0], t_x_bounds[1]):
                            for t_x_opt in t_x_opts: 
                                # timelag merged inps
                                # out: lvl2 inp/exp
                                x.timelag_inps('MO', t_x, t_x_opt)
                                print(x.nearby_NRFA)
                                # skip if inp empty 
                                # (~when t-x=0 and only 1 inp(=exp))
                                if x.inp.empty:
                                    print('skipping empty inp')
                                    continue
                                
                                # fit xgb for feature imps subsetting    
                                #
                                x.set_scale_inps(-1)  
                                x.xgb_model_fit()
                                x.merge_timelag_inps_subset(merge_opt, 20)
                                
                                for run in range(runs):
                                    pars.append([station_id, merge_opt, range_radius, ratio, t_x, t_x_opt, run])
         
                                    # train ml model
                                    x.set_scale_inps(-1)  
                                    x.keras_model(lr)
                                    x.keras_fit(ep)
                                    # x.model.save('_models/'+str(station_id)+'/mod'+str(run)+'.h5')
                                    
                                    # x.xgb_model_fit()
                                    
                                    # save fit/error
                                    big_RMSE.append(x.RMSE.values)
                                    print([station_id, merge_opt, range_radius, ratio, t_x, t_x_opt, run])
                                    print(x.RMSE.values[0][:2], '\n')
                    
    
    y = pd.DataFrame(np.concatenate(big_RMSE)).drop_duplicates()
    y.columns = ['cal', 'val', 'rows', 'cols']
    
    y_ = pd.merge(y, pd.DataFrame(pars, columns=['station', 'merge_opt', 'range', 'ratio', 't_x', 'opt', 'run']),
                  left_index=True, right_index=True).sort_values('val')
    y_.to_csv('GS/keras_RMSE_opts_big.csv')
      




gridsearch([23011],
           ['updwn', 'radius'],
           [10000, 40001, 10000],
           ['MO', 'NRFA_only'],
           [0.9, 0.91, 0.1],
           [0, 5],
           ['all', 'EA&station', 'NRFA'],
           3,
           lr=0.0001,
           ep=1)



'''
, 28015, 28022, 28044, 29002, 30002, 32008, 33013, 33066, 
            34010, 34012, 34018, 35003, 37008, 38003, 38029, 39001, 39026,
            39056, 39065, 39125, 40017, 45001, 46005, 46014, 47006, 47019,
            48001, 49006, 52010, 54017, 54057, 54110, 75017, 76017, 76021
'''

