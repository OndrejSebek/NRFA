import pandas as pd
import os

'''
o  log transform inps
o  split models for high/low flows

'''

import NRFA_v3 as nrfa
import QC_utils as qc_u
import kernets as knets

IDS = [46014]#[49006, 39125, 34010, 33013]#[76017, 54057, 39001]#[54110, 47019]
#[28022, 23011, 75017, 29002, 34012, 39026, 32008, 45001, 39065,
      # 52010, 38003, 47006, 33066, 76021, 35003, 37008, 38029, 46014] #

''' _______________________  STATION ID  _______________________ '''
for sta_id in IDS:
    st_id = sta_id
    
    # init stations
    x = nrfa.NRFA(st_id)
    x.set_ids_radius(20000, 20000, 20000)
    
    # adjust to missing gdf-live data
    x.update_ids_local(empty_NHA=0)
    
    if st_id == 35003:
        x.nearby_gauges_NHA.remove('LALDERR')
            

    ''' ______________ FILES ______________ '''
    
    # level1
    qc_u.fetch_NRFA_local_2019(st_id)
    #x.fetch_NRFA()

    x.fetch_agg_EA()
    x.fetch_MO()
    

    ratio = .9


    ''' ______________ xgb ______________ '''
    # do scaler separately (i=10)
    
    # level2
    x.merge_inps('NR', ratio)
    x.timelag_inps('MO', 3, 'all')
    x.set_scale_inps(99) 

    x.xgb_model_fit()
    x.xgb_plots()
    x.RMSE.to_csv('RMSEs/xgb_RMSE_'+str(st_id)+'.csv')
    x.xgb_reg.save_model('_models/'+str(st_id)+'/xgb.model')


    ''' ______________ keras ______________ '''
    
    # HIGHs
    
    # subset based on XGB feature importance (gain)
    x.timelag_inps_subset(20)

    flags = x.inp[x.xgb_feature_imps.iloc[0].colname]
    flags_thr = flags.median()
    flags = flags > flags_thr
    
    x.inp = x.inp[flags]
    x.exp = x.exp[flags]

    big_RMSE = []
    for i in range(10):
        x.set_scale_inps(i) 
    
        x.keras_model(.0001)
        x.keras_fit(10000)
        
        x.model.save('_models/'+str(st_id)+'/mod'+str(i)+'.h5')
        big_RMSE.append(x.RMSE.values)
        
        print(i, '/19: ', x.RMSE.values)

    # LOWs
        
    # subset based on XGB feature importance (gain)
    x.timelag_inps_subset(20)
    
    x.inp = x.inp[~flags]
    x.exp = x.exp[~flags]

    for i in range(10, 20):
        x.set_scale_inps(i) 
    
        x.keras_model(.0001)
        x.keras_fit(10000)
        
        x.model.save('_models/'+str(st_id)+'/mod'+str(i)+'.h5')
        big_RMSE.append(x.RMSE.values)
        
        print(i, '/19: ', x.RMSE.values)
        
        
    y = nrfa.pd.DataFrame(nrfa.np.concatenate(big_RMSE)).drop_duplicates()
    y.columns = ['cal', 'val', 'epoch', 'rows', 'cols']
    y.to_csv('RMSEs/keras_RMSE_'+str(st_id)+'.csv')
    
    x.keras_plots()

''' ____________________________ Kernets _______________________________ '''

for sta_id in IDS:
    x = knets.Kernets(sta_id, 20)
    
    x.nets = x.nets[:10]
    x.scalers = x.scalers[:10]
    x.net_rmses = x.net_rmses[:10]
    
    # divide low/high
    x.inps = x.inps[x.inps['46013'] > flags_thr]
    x.get_pred(bounds=0, conf=.95)
    
    highs = pd.DataFrame(x.m.copy(), index=x.inps.loc[flags].index,
                         columns=['m_h'])
    high_stds = pd.DataFrame(x.std.copy(), index=x.inps.loc[flags].index,
                             columns=['m_h_std'])
    
    x = knets.Kernets(sta_id, 20)
    
    x.nets = x.nets[10:]
    x.scalers = x.scalers[10:]
    x.net_rmses = x.net_rmses[10:]
    
    x.inps = x.inps[x.inps['46013'] < flags_thr]
    x.get_pred(bounds=0, conf=.95)
    
    lows = pd.DataFrame(x.m.copy(), index=x.inps.loc[~flags].index,
                        columns=['m_l'])
    low_stds = pd.DataFrame(x.std.copy(), index=x.inps.loc[~flags].index,
                             columns=['m_l_std'])
    
    
    
out1 = pd.merge(highs, lows, left_index = True, right_index=True, how='outer')    
out1['m_h'] = out1['m_h'].fillna(out1['m_l'])     
out1 = out1.dropna(axis=1) 
out1.columns=['nn_m']

out2 = pd.merge(high_stds, low_stds, left_index = True, right_index=True, how='outer')    
out2['m_h_std'] = out2['m_h_std'].fillna(out2['m_l_std'])     
out2 = out2.dropna(axis=1) 
out2.columns=['nn_std']

out = pd.merge(x.obs, out1, left_index = True, right_index=True)
out = pd.merge(out, out2, left_index = True, right_index=True)
out.to_csv('data/level3/'+x.station_id+'/'+x.station_id+'_merged.csv')  

    # x.save_pred()
    
    ''' preqc/qc files'''
    qc_u.fetch_preqc_qc(sta_id)
    mrgd = qc_u.merge_preqc_qc_nn(sta_id)
    
    # x.get_orig_exp()
    # x.find_outliers(1, .1) 
    # x.plots(2000)










