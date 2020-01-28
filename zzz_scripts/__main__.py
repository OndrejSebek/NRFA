import NRFA_v2 as nrfa
import QC_utils as qc_u
import kernets as knets


''' TODO '''
'''

try 33027, 33035 

38029
'''




''' _______________________  TRAIN & SAVE MODELS  _______________________ '''
for sta_id in [46014]:
    st_id = sta_id
    
    # init stations
    x = nrfa.NRFA(st_id)
    x.set_ids_radius(20000, 20000)
    
    # adjust to missing gdf-live data
    if st_id == 37008:
        x.nearby_NRFA = ['37031', '37015', '37014',
                         '37010', '37009', '37008',
                         '37006', '37003']
    elif st_id == 35003:
        x.nearby_NRFA.remove('35013')
        x.nearby_gauges_NHA.remove('LALDERR')
    elif st_id == 46014:
        x.nearby_NRFA.remove('46002')
    elif st_id == 38029:
        x.nearby_NRFA = ['33040', '38001', '38002', '38003',
                         '38004', '38007', '38011', '38012',
                         '38016', '38018', '38026', '38027',
                         '38028', '38029', '38030', '38031']
        x.nearby_gauges_NHA = []
        
    #x.fetch_NRFA()
    x.fetch_agg_EA()
    
    
    # ens
    big_RMSE = []
    for i in range(10):
        x.merge_inps(.8)
        x.timelag_inps(2, 'NRFA')
        
        x.set_scale_inps(i) 
    
        x.keras_model(.0001)
        x.keras_fit(10000)
        x.model.save('_models/'+str(st_id)+'/mod'+str(i)+'.h5')
        big_RMSE.append(x.RMSE.values)
         
           
    y = nrfa.pd.DataFrame(nrfa.np.concatenate(big_RMSE)).drop_duplicates()
    y.columns = ['cal', 'val', 'epoch', 'rows', 'cols']
    y.to_csv('RMSEs/keras_RMSE_'+str(st_id)+'.csv')
    
    
    
    x.keras_plots()


    ''' xgb '''
    # do scaler separately (i=10)
    x.merge_inps(.8)
    x.timelag_inps(3, 'NRFA')
    x.set_scale_inps(10) 

    x.xgb_model_fit()
    x.xgb_plots()
    x.xgb_reg.save_model('_models/'+str(st_id)+'/xgb.model')


''' ____________________________________________________________________ '''




        
''' ____________________________ Kernets _______________________________ '''
        
        
x = knets.Kernets(46014)

x.get_orig_exp()
x.get_pred(bounds=0, conf=.95)
x.save_pred()

x.find_outliers(1, .1)

x.plots(1000)


# x_grid = np.linspace(-1, 1)
# plt.plot(x_grid, x.kdes[0].evaluate(x_grid))

''' ___________________________________________________________________ '''





''' qc utils data process '''
''' _____________________ '''

st_id = 38029
n_dt = 300


# 1
qc_u.fetch_NRFA_local_2019(st_id)
qc_u.fetch_preqc_qc(st_id)


# 2
# move inp files -> @kernets


# 3
mrgd = qc_u.merge_preqc_qc_nn(st_id)
rmse, mae = qc_u.get_metrics(st_id, mrgd, n_dt)


# plots
qc_u.plot_preqc_qc_nn(st_id, mrgd, 1000)


''' _________ '''








''' ____ +++ _____ '''

# nrfa.gridsearch(station_id=46014,
#                 range_bounds=[2e4, 3e4, 1e4],
#                 ratio_bounds=[.6, .7, .2],
#                 t_x_bounds=[1, 2],
#                 t_x_opts=['NRFA', 'all', 'EA&station', 'station', 'EA'],
#                 runs=1) 









































