import NRFA_v3 as nrfa
import kernets as knets
import QC_utils as qc_u

IDS = [34018, 30002, 28044, 48001, 40017, 46005, 54017, 28015, 39056, 46014,
       49006, 39125, 34010, 33013, 76017, 54057, 39001, 54110, 47019, 28022, 
       23011, 75017, 29002, 34012, 39026, 32008, 45001, 39065, 52010, 38003, 
       47006, 33066, 76021, 35003, 37008, 38029] 

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
    x.timelag_inps_subset(20)

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



