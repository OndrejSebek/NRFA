import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt

import os




def fetch_NRFA_local_2019(station_id):
    bigf = pd.DataFrame()
    for file in os.listdir('data/nrfa_raw/'+str(station_id)):
        print(file)
        curf = pd.read_csv('data/nrfa_raw/'+str(station_id)+'/'+file,
                           index_col=0, header=None)
        
        # variable meta rows
        if curf[1].iloc[19][0].isnumeric():
            skiprows = 18
        else:
            skiprows = 19
            
        curf = curf.iloc[skiprows:]    
        curf = curf[[1]].drop(curf.index[0], axis=0)
        
        # df formating
        curf.index.name = 'date'
        curf.columns=[file[5:10]]
        
        # date format
        dtformat = curf.index[0]
        
        if dtformat[4] == '-':
            dtf = '%Y-%m-%d'
        else:
            dtf = '%d/%m/%Y'        
        
        curf.index = pd.to_datetime(curf.index, format=dtf).normalize()
        
        # MERGE to bigf
        if bigf.empty:
            bigf = curf.copy()
        else:
            bigf = pd.merge(bigf, curf,
                            left_index=True, right_index=True,
                            how='outer')
            
    if not os.path.exists('data/level1/'+str(station_id)):
        os.mkdir('data/level1/'+str(station_id))

    #bigf.to_csv('data/_NRFA_qc/data/'+str(station_id)+'/'+str(station_id)+'_NRFA.csv')
    bigf = bigf.sort_index()
    bigf.to_csv('data/level1/'+str(station_id)+'/'+str(station_id)+'_NRFA.csv')



def fetch_preqc_qc(station_id):
    # y1 = pd.read_csv('data/_NRFA_qc/data/'+str(station_id)+'/'+str(station_id)+'_NRFA.csv',
    #                  index_col=0)[[str(station_id)]]
    y1 = pd.read_csv('data/level1/'+str(station_id)+'/'+str(station_id)+'_NRFA.csv',
                     index_col=0)[[str(station_id)]]

    y2 = pd.read_csv('meta/_NRFA_qc/gdf-live-audit-counts.csv', index_col=1)
    y2 = y2[y2['STATION'] == station_id][['FLOW_VALUES']]
    y2.index = pd.to_datetime(y2.index, format='%Y-%m-%d %H:%M:%S').normalize()


    mrgd = pd.merge(y1, y2, left_index=True, right_index=True)
    
    mrgd['orig'] = mrgd['FLOW_VALUES'].apply(get_orig)
    mrgd['new'] = mrgd['FLOW_VALUES'].apply(get_new)


    ''' change exp to pre-audit '''
    
    # exp_o = pd.read_csv('data/_NRFA_qc/data/'+str(station_id)+'/'+str(station_id)+'_exp_merged.csv',
    #                     index_col=0)
    exp_o = pd.read_csv('data/level2/'+str(station_id)+'/'+str(station_id)+'_exp_merged.csv',
                        index_col=0)
    
    mrgd_exp = pd.merge(exp_o, mrgd[['orig']],
                        left_index=True, right_index=True)
    
    mrgd_exp.loc[mrgd_exp['orig'] == 'nan', ['orig']] = np.nan
    mrgd_exp['orig'] = mrgd_exp['orig'].fillna(mrgd_exp[str(station_id)]).astype(float)
    
    
    ''' export preqc/qcd '''
    if not os.path.exists('data/level3/'+str(station_id)+'/comp'):
        os.mkdir('data/level3/'+str(station_id)+'/comp')

    mrgd_exp[['orig']].to_csv('data/level3/'+str(station_id)+'/comp/'+str(station_id)+'_orig.csv')
    mrgd_exp[[str(station_id)]].to_csv('data/level3/'+str(station_id)+'/comp/'+str(station_id)+'_qcd.csv')



def merge_preqc_qc_nn(station_id):
    # read qcd data
    mrgd = pd.read_csv('data/level3/'+str(station_id)+'/'+str(station_id)+'_merged.csv',
                       index_col=0)
    # read pre-qc data
    obs_orig = pd.read_csv('data/level3/'+str(station_id)+'/comp/'+str(station_id)+'_orig.csv', index_col=0)

    # merge and return
    out = pd.merge(mrgd, obs_orig, left_index=True, right_index=True, how='outer')
    out['orig'] = out['orig'].fillna(out[str(station_id)])
    
    out.to_csv('data/level3/'+str(station_id)+'/comp/'+str(station_id)+'_merged.csv')
    return out



def get_metrics(station_id, mrgd_orig, n_dt):
    rmse_1 = sqrt(mean_squared_error(mrgd_orig[str(station_id)].values[-n_dt:], mrgd_orig['orig'].values[-n_dt:])) 
    rmse_2 = sqrt(mean_squared_error(mrgd_orig[str(station_id)].values[-n_dt:], mrgd_orig['nn_m'].values[-n_dt:])) 
    rmse_3 = sqrt(mean_squared_error(mrgd_orig['orig'].values[-n_dt:], mrgd_orig['nn_m'].values[-n_dt:])) 
    
    mae_1 = np.mean(abs(mrgd_orig[str(station_id)].values[-n_dt:] - mrgd_orig['orig'].values[-n_dt:]))
    mae_2 = np.mean(abs(mrgd_orig[str(station_id)].values[-n_dt:] - mrgd_orig['nn_m'].values[-n_dt:]))
    mae_3 = np.mean(abs(mrgd_orig['orig'].values[-n_dt:] - mrgd_orig['nn_m'].values[-n_dt:]))
    
    rmse = [rmse_1, rmse_2, rmse_3]
    mae = [mae_1, mae_2, mae_3]

    return rmse, mae



''' ________________ helpers __________________ '''

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_orig(x):
    x_ = str(x)[:4]
    
    if x_ != 'nan':
        for i in range(len(x_)):
            if not is_float(x_):
                x_ = x_[:-1]        
    return x_
    
def get_new(x):
    x_ = str(x)[-5:]

    if x_ != 'nan':
        for i in range(len(x_)):
            if not is_float(x_):
                x_ = x_[1:]           
    return x_

''' ____________________________________________ '''


# REDUNDANT
def plot_preqc_qc_nn(station_id, mrgd, n_dt):
    st_id = str(station_id)
    x_m = pd.read_csv('data/level3/'+st_id+'/nn/x_m.csv', index_col=0)
    x_m.index = mrgd.index
    x_l = pd.read_csv('data/level3/'+st_id+'/nn/x_l.csv', index_col=0)
    x_l.index = mrgd.index
    x_h = pd.read_csv('data/level3/'+st_id+'/nn/x_h.csv', index_col=0)
    x_h.index = mrgd.index

    if n_dt == 0:
        n_dt = mrgd.shape[0]
        
    fig, ax = plt.subplots(2, 1, figsize=(30,10), dpi=600, gridspec_kw={'height_ratios': [3, 1]}, sharey=False)
    ax[0].plot(mrgd['orig'].values[-n_dt:], label='pre-qc', c='darkcyan')
    ax[0].plot(np.concatenate(x_m[-n_dt:].values), label='mod', c='red')
    ax[0].plot(mrgd[str(st_id)].values[-n_dt:], label='qcd', c='black')
    ax[0].legend(loc=1)
    
    ax[1].plot(np.concatenate(x_m[-n_dt:].values)-mrgd['orig'].values[-n_dt:], label='mod_m', c='red')
    ax[1].plot(mrgd['orig'].values[-n_dt:]-mrgd['orig'].values[-n_dt:], label='preqc', c='darkcyan')
    ax[1].plot(np.concatenate(x_l[-n_dt:].values)-mrgd['orig'].values[-n_dt:], label='mod_l', c='salmon')
    ax[1].plot(np.concatenate(x_h[-n_dt:].values)-mrgd['orig'].values[-n_dt:], label='mod_h', c='salmon')
    ax[1].plot(mrgd[str(st_id)].values[-n_dt:]-mrgd['orig'].values[-n_dt:], label='qcd', c='black')
    ax[1].legend(loc=1)
    
    if not os.path.exists('plots/'+str(station_id)+'/comp'):
        os.mkdir('plots/'+str(station_id)+'/comp')

    plt.savefig('plots/'+str(station_id)+'/comp/comp.jpg')
    plt.close()



# calculate overall and Q70/30 fit stats
#   (RMSEs, stds, NSEs)
#
def comp_Qn_nRMSE():
    big_rmse = []
    for station in os.listdir('data/level3/'):
        cur_dt = pd.read_csv('data/level3/'+station+'/'+station+'_merged.csv',
                             index_col=0).sort_values(station, ascending=True)
        
        _q70 = int(cur_dt.shape[0]*.3)
        _q30 = int(cur_dt.shape[0]*.7)
        
        # normalisation parameter (x_mean)
        station_obs_mean = cur_dt[station].mean()
        
        rmses = [station, sqrt(mean_squared_error(cur_dt[station], cur_dt['nn_m']))/station_obs_mean]  
        stds = [cur_dt['nn_std'].mean()/station_obs_mean]
        NSEs = [1 - ( ((cur_dt[station]-cur_dt['nn_m'])*(cur_dt[station]-cur_dt['nn_m'])).sum()
                     /((cur_dt[station]-cur_dt['nn_m'].mean())*(cur_dt[station]-cur_dt['nn_m'].mean())).sum()
                     )]
        
        bounds = [0, _q70, _q30, cur_dt.shape[0]]
        for i in range(3):
            c_dt = cur_dt[bounds[i]:bounds[i+1]]
        
            c_rmse = (sqrt(mean_squared_error(c_dt[station],
                                              c_dt['nn_m']))
                      )/station_obs_mean    
            
            rmses.append(c_rmse)
            stds.append(c_dt['nn_std'].mean()/station_obs_mean)
            
        rmses.extend(stds)
        rmses.extend(NSEs)

        big_rmse.append(rmses)
        
    big_rmse = pd.DataFrame(big_rmse, columns=['station', 'comb_nRMSE',
                                               '<q70_nRMSE', 'q30-q70_nRMSE', '>q30_nRMSE',
                                               'comb_nSTD',
                                               '<q70_nSTD', 'q30-q70_nSTD', '>q30_nSTD',
                                               'comb_NSE'])
    
    big_rmse.to_csv('meta/comp/Qn_stats.csv', index=False)



def comp_v2_v3_RMSEs():
    big_df = []
    for file in os.listdir('plots'):
        v2_rmse = pd.read_csv('../NRFA_.2/RMSEs/keras_RMSE_'+file+'.csv',
                             index_col=0)[['cal', 'val']]
        v3_rmse = pd.read_csv('RMSEs/keras_RMSE_'+file+'.csv',
                             index_col=0)[['cal', 'val']]
        print(v3_rmse.mean().values, v2_rmse.mean().values)
        ratio = (v3_rmse.mean().values-v2_rmse.mean().values)/v2_rmse.mean().values*100
        
        big_df.append([file, ratio[0], ratio[1]])  
    
    big_df = pd.DataFrame(big_df, columns=['station', 'cal', 'val'])
    big_df.to_csv('meta/comp/v2_v3_RMSEs.csv', index=False)






def gdf_qc_changes():
    meta = pd.read_csv('meta/_NRFA_qc/gdf-live-audit-counts.csv')

    big = []
    for station in meta['STATION'].unique():
        sub = meta[meta['STATION'] == station]
        
        cur_preqc = sub['FLOW_VALUES'].apply(get_orig).astype(float)
        cur_qcd = sub['FLOW_VALUES'].apply(get_new).astype(float)

        # normalisation factor
        mean_obs = cur_qcd.mean()
        
        # comp change
        ch_sum = abs(cur_qcd-cur_preqc).sum()/mean_obs     
        ch_max = abs(cur_qcd-cur_preqc).max()/mean_obs
        
        ch_a = abs(cur_qcd-cur_preqc).sort_values(ascending=False)[:10].sum()/mean_obs

        big.append([station, ch_sum, ch_a, ch_max])
    
    big = pd.DataFrame(big, columns=['station', 'all_sum', 'top10_sum', 'all_max'])
    big.to_csv('meta/qc_changes.csv', index=False)


def model_fit_inp_tables(station, n_inps):
    model_inps = pd.read_csv('_model_inps/'+station+'.csv')
    
    stations = model_inps['colname'][:n_inps].astype(str)
    
    sts = []
    for i in stations:
        if i[-2] == '-':
            st = i[:-2]
        else:
            st = i
            
        sts.append(st)
    sts = np.unique(sts)
    
    st_map = pd.DataFrame(np.zeros((len(sts), 4)),
                          index=sts, columns=range(4))
    
    for i in stations:
        if i[-2] == '-':
            st = i[:-2]
            t = i[-1]
        else:
            st = i
            t = 0
    
        st_map.loc[st, int(t)] = 1
   
    st_map.loc[station, 0] = 2
    
    st_map.to_csv('_model_inps_subtable/'+station+'.csv', index=True)








for st in os.listdir('plots'):
    model_fit_inp_tables(st, 20)
    













