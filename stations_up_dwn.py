import pandas as pd


dt = pd.read_excel('stations_upstream_downstream/nrfa_nearest_sites.xlsx',
                   sheet_name='nrfa_nearest_sites')


station_id = 39046
sub_dt = dt[dt['station'] == station_id]
stationz = list(sub_dt['related_station'].astype(str))   # + [str(station_id)]



import NRFA_v3 as nrfa
import kernets as knets
import QC_utils as qc_u

IDS = ['39046']

'''
[23011, 28015, 28022, 28044, 29002, 30002, 32008, 33013, 33066,
34010, 34012, 34018, 35003, 37008, 38003, 38029, 39001, 39026,
39056, 39065, 39125, 40017, 45001, 46005, 46014, 47006, 47019,
48001, 49006, 52010, 54017, 54057, 54110, 75017, 76017, 76021]

'''


''' _______________________  STATION ID  _______________________ '''
for st_id in IDS:
    
    # init stations
    x = nrfa.NRFA(st_id)
    x.set_ids_radius(20000, 20000, 20000)
    
    # set/adjust to missing gdf-live data
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
    # do scaler separately (i=99)
    
    # level2
    x.merge_inps('NRFA_only', ratio)
    x.timelag_inps('MO', 3, 'all')
    x.set_scale_inps(99) 

    x.xgb_model_fit()
    x.xgb_plots()
    x.RMSE.to_csv('RMSEs/xgb_RMSE_'+str(st_id)+'.csv')
    x.xgb_reg.save_model('_models/'+str(st_id)+'/xgb.model')


    ''' ______________ keras ______________ '''
    
    
    # subset based on XGB feature importance (gain)
        # x.merge_inps('NRFA_only', ratio)
    x.merge_timelag_inps_subset('NRFA_only', 20)

    big_RMSE = []
    for i in range(20):
        # cal/val split
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

for st_id in IDS:
    x = knets.Kernets(st_id, 10)
    
    x.get_pred(bounds=0, conf=.95)
    x.save_pred()
    
    ''' preqc/qc files'''
    qc_u.fetch_preqc_qc(st_id)
    mrgd = qc_u.merge_preqc_qc_nn(st_id)
    
    # x.get_orig_exp()
    # x.find_outliers(1, .1) 
    # x.plots(2000)










m = 0
s = ''
for stat in dt['station'].unique():
    
    c_m = dt[dt['station'] == stat].shape[0]
    
    if c_m > m:
        m = c_m
        s = stat
    
    
    
    
