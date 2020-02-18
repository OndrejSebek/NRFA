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

:::     + test resampled bagging against current
        + error (corection) correlation

'''

''' __________________________  STATION ID  ______________________________ '''

IDS = [49006]

def model_trainer(IDS, dist,
                  RG_src='MO', inp_ratio=.95,
                  timelag_opt='all', timelag_t_x=5, 
                  n_subset_inp=30, tt_split=400, cv_split=.25,
                  n_models=20, lr=1e-4, ep=1e5):
    """
    Trains an ensemble of NN models.
    
    I:      determines inp features, creates dataset to model
    II:     fits XGB model - benchmark, feature importance (subsetting)
    III:    trains an ensemble of NNs on the train split of the current 
            dataset, 
            
    Exports trained NN and XGB models, inp data scalers, fit/error stats,
    plots.
    
    Parameters
    ----------
    IDS : list [int/str]
        NRFA station IDs.
    dist : int
        Distance from target NRFA station
    RG_src : str, optional
        Input rainfall data src. The default is 'MO'.
    inp_ratio : float <0, 1>, optional
        Data completeness station selection parameter. The default is .95.
    timelag_opt : str, optional
        Which features to timelag. The default is 'all'.
    timelag_t_x : int, optional
        Timelagging nr. of days. The default is 5.
    n_subset_inp : int, optional
        Subsetting nr. of most important inp features. The default is 30.
    tt_split : int, optional
        Nr. of days to leave out for test phase. The default is 400.
    cv_split : float <0, 1>, optional
        Ratio of cal/val split. The default is .25.
    n_models : int, optional
        Nr. of models to train. The default is 20.
    lr : float, optional
        NN earning rate. The default is 1e-4.
    ep : int, optional
        NN training epoch threshold. The default is 1e5. (x early stopping)

    """
    for st_id in IDS:
        
        # init stations
        x = nrfa.NRFA(st_id)
        x.set_ids_radius(dist, dist, dist)
        
        # adjust to missing gdf-live data
        x.update_ids_local(empty_NHA=0)
        
        if st_id == 35003:
            x.nearby_gauges_NHA.remove('LALDERR')
    
        ''' ___________________________ files ____________________________ '''
        
        # level1 data
        qc_u.fetch_NRFA_local_2019(st_id)
        #x.fetch_NRFA()
        # x.fetch_agg_EA()
        x.fetch_MO()
    
        ''' ____________________________ xgb _____________________________ '''
        
        # level2 data
        x.merge_inps(ratio=inp_ratio)
        x.timelag_inps(timelag_t_x, timelag_opt, RG_src)
        x.scale_split_traintest(n_traintest_split=0,
                                ratio_calval_split=cv_split) 
    
        x.xgb_model_fit()
        x.xgb_plots()
        x.RMSE().to_csv('RMSEs/xgb_RMSE_'+str(st_id)+'.csv')
        x.xgb_reg.save_model('_models/'+str(st_id)+'/xgb.model')
    
        ''' __________________________ keras _____________________________ '''
        
        # subset based on XGB feature importance (gain)
        x.merge_timelag_inps_subset(n_subset_inp, RG_src)
    
        big_RMSE = []
        for i in range(n_models):
            # cal/val split
            x.scale_split_traintest(n_traintest_split=tt_split,
                                    ratio_calval_split=cv_split) 
            
            # train NNs
            x.keras_model(lr)
            x.keras_fit(ep)
            
            # export model & scaler, append current fit stats
            x.save_model(i)
            big_RMSE.append(x.RMSE().values)
            
            # console print
            print(i, '/19: ', x.RMSE().values)
        
        # concat fit stats into pd.df & format
        y = nrfa.pd.DataFrame(nrfa.np.concatenate(big_RMSE)).drop_duplicates()
        
        if not x.test_split:
            y.columns = ['cal', 'val', 'rows', 'cols']
        else:
            y.columns = ['cal', 'val', 'test', 'rows', 'cols']
        
        # export fit & plots
        y.to_csv('RMSEs/keras_RMSE_'+str(st_id)+'.csv')
        x.keras_plots()


model_trainer([49006], 20, ep=1)

''' _____________________________ KERNETS ________________________________ '''

for st_id in IDS:
    z = knets.Kernets(st_id, 10)
    
    # preqc/qc files
    qc_u.fetch_preqc_qc(st_id)

    # preds
    z.get_mod(bounds=0, conf=.95)
    z.save_mod_merged()
    
    # z.get_orig_exp()
    # z.find_outliers() 
    # z.plots(2000)


# mrgd.iloc[-400:].plot(figsize=(100, 20))
# plt.savefig('awda.png')
# plt.legend()
# plt.close()
