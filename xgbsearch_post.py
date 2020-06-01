import pandas as pd
import matplotlib.pyplot as plt
import os



''' ___________________________ PLOTS ___________________________________ '''

def xgbsearch_fit_sep(station_id, metric):
    """
    Plot fit stats for xgbsearched par combinations for each station 
    separately.

    """
    dt = pd.read_csv(f'GS/xgbsearch_{station_id}.csv', index_col=1).drop('Unnamed: 0', axis=1)
    
    # sub = dt.loc[:, dt.loc['station'] == c_station]
    # sub = sub.dropna(thresh=1, axis=0)
    
    asc = True if metric == 'NSE' else False
    dt = dt.sort_values(f'{metric}_val_NN_avg', axis=1,
                        ascending=asc)
    
    ax = dt.loc[[f'{metric}_cal_NN_avg', f'{metric}_val_NN_avg']].astype(float).T.plot.bar(figsize=(25, 20))
    # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
    ax.set_xticklabels(dt.loc['inp_opt'].str[:] + ' ' + dt.loc['range_opt'].str[:]
                       + ' ' + dt.loc['range_dist'].str[:2] + 'km ' 
                       + dt.loc['cols_sub'].str[:] + 'inp')
    plt.savefig(f'GS/plots/{station_id}_{metric}.png')
    plt.close()


def xgbsearch_fit_comb(metric):
    """
    Plot averaged fit stats for xgbsearched par combinations for all stations
    combined.

    """
    header_vars = ['station', 'range_opt', 'range_dist', 'inp_opt', 'cols_sub']
    
    big = pd.DataFrame()
    
    for file in os.listdir('GS/'):
        if file[-4:] == '.csv':
            station_id = file[10:15]
            print(station_id)
        else:
            continue
        
        dt = pd.read_csv(f'GS/xgbsearch_{station_id}.csv',
                         index_col=1).drop('Unnamed: 0', axis=1)
    
        headers = dt.loc[header_vars]
        vals = dt.iloc[-4:].astype(float)   
        
        headers.columns = range(headers.shape[1])
        vals.columns = range(vals.shape[1])
        
        if big.empty:
            big = vals.copy()
        else:
            big = big + vals
    
    big.iloc[-4:] = big.iloc[-4:]/len(dt.loc['station'].unique())
    big = headers.append(big)

    asc = True if metric == 'NSE' else False
    big = big.sort_values(f'{metric}_val_NN_avg', axis=1,
                          ascending=asc)

    ax = big.loc[[f'{metric}_cal_NN_avg', f'{metric}_val_NN_avg']].astype(float).T.plot.bar(figsize=(25, 20))
    # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
    ax.set_xticklabels(big.loc['inp_opt'].str[:] + ' ' + big.loc['range_opt'].str[:]
                       + ' ' + big.loc['range_dist'].str[:2] + 'km ' 
                       + big.loc['cols_sub'].str[:] + 'inp')
    plt.savefig(f'GS/plots/big_{metric}.png')
    plt.close()
    
  
def presentation_plot_fit():
    """
    Plot NSE bar plot of NN ens fit.

    """
    dt = pd.read_csv('meta/comp/Qn_stats.csv', index_col=0)
    dt = dt[[dt.columns[-1]]]
    
    # fig = go.Figure(go.Heatmap(x=dt.columns.astype(str),
    #                            y=dt.index.astype(str).values+':',
    #                            z=dt.values,
    #                             zmin=0, zmax=1,
    #                            colorscale='YlOrRd',
    #                            name='st inps'))
    
    # # fig['data'][0]['showscale'] = True
    # fig['layout']['xaxis'].update(side='top')
    
    # # update plot background color to transparent
    # # fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)',
    # #                      margin_l=0)
    # # fig.write_image("fig1.png")   
    # pio.write_html(fig, file='fig1.html', auto_open=False)
    
    dt.sort_values('comb_NSE').plot.bar(figsize=(10,5))    
    plt.savefig('GS/NSE_fit.png')
    plt.close()


def presentation_plot_fit_xgb():
    """
    NOT USED: Plot NSE bar plot of XGB models

    """
    import xgboost as xgb
    import joblib
    
    for station in os.listdir('_models'):
        if station == '39046':
            continue
        
        xgb_reg = xgb.XGBRegressor()
        xgb_reg.load_model(f'_models/{station}/xgb.model')
        # print(station, 'loaded (xgb)')
        
        scaler_inp = joblib.load(f'_models/{station}/scaler99.pkl')
        
        # data
        c_inp = pd.read_csv(f'data/level2/{station}/{station}_inp.csv',
                            index_col=0)
        c_exp = pd.read_csv(f'data/level2/{station}/{station}_exp.csv',
                            index_col=0).values[:, 0]
        
        if len(c_exp) == 0:
            continue
        
        # station_obs_mean = c_exp.mean()
        
        c_inp_t = scaler_inp.fit_transform(c_inp)
        c_mod = xgb_reg.predict(c_inp_t)
        
        NSE = 1 - ( ((c_exp-c_mod)*(c_exp-c_mod)).sum()
                        /((c_exp-c_exp.mean())*(c_exp-c_exp.mean())).sum()
                        )
        print(NSE)
        
        # print(c_exp, c_mod)


def presentation_plot_fit_xgb_retrain():
    """
    Plot NSE bar plot of XGB model fit. 

    """
    import xgboost as xgb
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    
    big = []
    for station in os.listdir('_models'):
        if station == '39046':
            continue
        
        # data
        c_inp = pd.read_csv(f'data/level2/{station}/{station}_inp.csv',
                            index_col=0)
        c_exp = pd.read_csv(f'data/level2/{station}/{station}_exp.csv',
                            index_col=0).values[:, 0]
        
        if len(c_exp) == 0:
            continue
        
        
        scaler_inp = preprocessing.StandardScaler()
        c_inp_t = scaler_inp.fit_transform(c_inp)
        
        x_cal, x_val, y_cal, y_val = train_test_split(c_inp_t, c_exp, 
                                                      test_size=0.3,
                                                      random_state=None, 
                                                      shuffle=True)
      
        
        xgb_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, 
                                colsample_bynode=1, colsample_bytree=1, gamma=0,
                                importance_type='gain', learning_rate=0.1, max_delta_step=0,
                                max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                                n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                silent=None, subsample=1, verbosity=1)
        
        
        xgb_reg.fit(x_cal, y_cal)        
        c_mod = xgb_reg.predict(c_inp_t)
        
        
        
        NSE = 1 - ( ((c_exp-c_mod)*(c_exp-c_mod)).sum()
                        /((c_exp-c_exp.mean())*(c_exp-c_exp.mean())).sum()
                        )
        big.append([station, NSE])
        
    NSEs = pd.DataFrame(big)
    NSEs = NSEs.set_index(0)
    NSEs.columns = ['xgb_NSE']
    
    NSEs.sort_values('xgb_NSE').plot.bar(figsize=(10,5))    
    plt.savefig('GS/NSE_fit_xgb.png')
    plt.close()

''' __________________________ / PLOTS __________________________________ '''



# for file in os.listdir('GS/'):
#     if file[-4:] == '.csv':
#         for opt in ['NSE', 'nRMSE']:
#             xgbsearch_fit_sep(file[10:15], opt)


# xgbsearch_fit_comb('nRMSE')





def postproc(path, metric, thr): # redundant
    header_vars = ['station', 'range_opt', 'range_dist', 'inp_opt', 'cols_sub']

    big = pd.DataFrame()
    for file in os.listdir(path):
        if file[-4:] == '.csv':
            station_id = file[10:15]
            print(station_id)
        else:
            print(f'skip {path}/{file}')
            continue
        
        dt = pd.read_csv(f'{path}/{file}',
                         index_col=1).drop('Unnamed: 0', axis=1)    
        
        headers = dt.loc[header_vars]
        vals = dt.iloc[-4:].astype(float)  
        print(headers)
        headers.columns = range(headers.shape[1])
        vals.columns = range(vals.shape[1])

        asc = True if metric == 'NSE' else False
        vals = vals.sort_values(f'{metric}_val_NN_avg', axis=1,
                                ascending=asc)
        
        if asc:
            thr_val = vals.loc[f'{metric}_val_NN_avg', vals.columns[-1]] - vals.loc[f'{metric}_val_NN_avg', vals.columns[-1]] * (thr)
            subset = list(vals.columns[vals.loc[f'{metric}_val_NN_avg'] > thr_val])
        else:
            thr_val = vals.loc[f'{metric}_val_NN_avg', vals.columns[-1]] + vals.loc[f'{metric}_val_NN_avg', vals.columns[-1]] * (thr) 
            subset = list(vals.columns[vals.loc[f'{metric}_val_NN_avg'] < thr_val])
        print(subset)
        
        headers = headers[subset]
        vals = vals[subset]
        
        big = pd.concat([big, headers], axis=1, ignore_index=True)
        
        ax = vals.loc[[f'{metric}_cal_NN_avg', f'{metric}_val_NN_avg']].astype(float).T.plot.bar(figsize=(25, 20))
        # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
        ax.set_xticklabels(headers.loc['inp_opt'].str[:] + ' ' + headers.loc['range_opt'].str[:]
                           + ' ' + headers.loc['range_dist'].str[:2] + 'km ' 
                           + headers.loc['cols_sub'].str[:] + 'inp')
        plt.savefig(f'GS/plots/{station_id}_{metric}.png')
        plt.close()
    
    big.to_csv(f'{path}/plots/{metric}_{thr}_pars.csv')
        
        
# postproc('GS', 'nRMSE', .1)


def postprc(metric):
    header_vars = ['range_opt', 'range_dist', 'inp_opt', 'cols_sub']
    
    big = pd.DataFrame()
    files = 0
    for file in os.listdir('GS/'):
        if file[-4:] == '.csv':
            station_id = file[10:15]
            print(station_id)
            files += 1
        else:
            continue
        
        dt = pd.read_csv(f'GS/xgbsearch_{station_id}.csv',
                         index_col=1).drop('Unnamed: 0', axis=1)
        
        headers = dt.loc[header_vars]
        vals = dt.iloc[-4:].astype(float)   
        
        headers.columns = range(headers.shape[1])
        vals.columns = range(vals.shape[1])
        
        asc = True if metric == 'NSE' else False
        vals = vals.sort_values(f'{metric}_val_NN_avg', axis=1,
                                ascending=asc)
        
        if asc:        
            base_cal = vals.loc[f'{metric}_cal_NN_avg'].max()
            base_val = vals.loc[f'{metric}_val_NN_avg'].max()
            
            vals.loc['perc_cal'] = (base_cal - vals.loc[f'{metric}_cal_NN_avg'])/base_cal
            vals.loc['perc_val'] = (base_val - vals.loc[f'{metric}_val_NN_avg'])/base_val
        else:
            base_cal = vals.loc[f'{metric}_cal_NN_avg'].min()
            base_val = vals.loc[f'{metric}_val_NN_avg'].min()

            vals.loc['perc_cal'] = (vals.loc[f'{metric}_cal_NN_avg'] - base_cal)/base_cal  
            vals.loc['perc_val'] = (vals.loc[f'{metric}_val_NN_avg'] - base_val)/base_val  
        
        if big.empty:
            big = vals.loc[['perc_cal', 'perc_val']].copy()
        else:
            big = big + vals.loc[['perc_cal', 'perc_val']]

    big /= files
    big = headers.append(big)
    
    big.sort_values('perc_val', axis=1, inplace=True, ascending=asc)
    print(big)
    big.to_csv('GS/plots/perc.csv')
    
    ax = big.loc[['perc_cal', 'perc_val']].astype(float).T.plot.bar(figsize=(25, 20))
    # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
    ax.set_xticklabels(big.loc['inp_opt'].str[:] + ' ' + big.loc['range_opt'].str[:]
                        + ' ' + big.loc['range_dist'].str[:2] + 'km ' 
                        + big.loc['cols_sub'].str[:] + 'inp')
    plt.savefig(f'GS/plots/perc.png')
    plt.close()


# postprc('nRMSE')



def get_GS_best_inps():
    for file in os.listdir("GS/"):
        if file[-4:] != '.csv':
            continue
        
        x = pd.read_csv(f"GS/{file}", index_col=0, dtype=str)
        
        x_b = x[[x.columns[0], x.columns[14]]]
        
        header_index = x_b[x_b['var'] == "station"].index[0]
        
        x_b = x_b.loc[:(header_index-1)]
        x_b = x_b[~x_b[x_b.columns[1]].isna()][["var"]]
        x_b.to_csv(f"_model_inps/{file}", index=False)

get_GS_best_inps()






# ___________________________________________________________________ #

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


def EA_east_north_site_list_format(): 
    x = pd.read_excel('EA_site_list.xlsx', header=None, dtype=str).drop_duplicates()
    x = pd.DataFrame(x[0].apply(format_EA_ids_helper))
    y = pd.read_csv('meta/COSMOS_meta_updated.csv', dtype=str)
    
    o = pd.merge(x, y, left_on=0, right_on='API_ID', how='outer', indicator=True).drop_duplicates()
    ok = o[o['_merge'] == 'both']
    l = o[o['_merge'] == 'left_only']
    
    out1 = ok[[0, 'easting', 'northing']].drop_duplicates()
    
    oo = pd.merge(l[[0]], y, left_on=0, right_on='NHA_ID', how='outer', indicator=True)
    
    ok = oo[oo['_merge'] == 'both']
    l = oo[oo['_merge'] == 'left_only']
    
    out2 = ok[[0, 'easting', 'northing']].drop_duplicates()
    
    out = pd.concat([out1, out2], ignore_index=True)
    out.columns = ['id', 'easting', 'northing']
    out.to_csv('EA_site_list_east_north.csv', index=False)