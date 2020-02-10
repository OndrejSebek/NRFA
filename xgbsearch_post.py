import pandas as pd
import matplotlib.pyplot as plt
import os

dt = pd.read_csv('GS/xgbsearch.csv', index_col=1).drop('Unnamed: 0', axis=1)


# ____________________________________________________________________

# plot fit stats for xgbsearched par combinations 
#   for each station separately
#
def xgbsearch_fit_sep():
    for c_station in dt.loc['station'].unique():
        sub = dt.loc[:, dt.loc['station'] == c_station]
        sub = sub.dropna(thresh=1, axis=0)
        
        sub = sub.sort_values('NSE_val_NN_avg', axis=1)
        
        ax = sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T.plot.bar(figsize=(25, 20))
        # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
        ax.set_xticklabels(sub.loc['inp_opt'].str[:] + ' ' + sub.loc['range_opt'].str[:]
                           + ' ' + sub.loc['range_dist'].str[:2] + 'km ' 
                           + sub.loc['cols_sub'].str[:] + 'inp')
        plt.savefig('GS/plots/'+c_station+'.png')
        plt.close()


# ____________________________________________________________________

# plot averaged fit stats for xgbsearched par combinations 
#   for all stations
#
def xgbsearch_fit_comb():
    header_vars = ['station', 'range_opt', 'range_dist', 'inp_opt', 'cols_sub']
    
    big = pd.DataFrame()
    
    for c_station in dt.loc['station'].unique():
        sub = dt.loc[:, dt.loc['station'] == c_station]
        sub = sub.dropna(thresh=1, axis=0)
    
        headers = sub.loc[header_vars]
        vals = sub.iloc[-4:].astype(float)   
        
        headers.columns = range(headers.shape[1])
        vals.columns = range(vals.shape[1])
        
        if big.empty:
            big = vals.copy()
        else:
            big = big + vals
    
    big.iloc[-4:] = big.iloc[-4:]/len(dt.loc['station'].unique())
    big = headers.append(big)
    
    
    big = big.sort_values('NSE_val_NN_avg', axis=1)
    
    ax = sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T.plot.bar(figsize=(25, 20))
    # plt.plot(sub.loc[['NSE_cal_NN_avg', 'NSE_val_NN_avg']].astype(float).T)
    ax.set_xticklabels(sub.loc['inp_opt'].str[:] + ' ' + sub.loc['range_opt'].str[:]
                       + ' ' + sub.loc['range_dist'].str[:2] + 'km ' 
                       + sub.loc['cols_sub'].str[:] + 'inp')
    plt.savefig('GS/plots/big.png')
    plt.close()
    

# ____________________________________________________________________

# plot NSE bar plot of NN ens
#    
def presentation_plot_fit():
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


# ____________________________________________________________________

# plot NSE bar plot of XGB models
#
#       : NOT USED
#
def presentation_plot_fit_xgb():
    import xgboost as xgb
    import joblib
    
    for station in os.listdir('_models'):
        if station == '39046':
            continue
        
        xgb_reg = xgb.XGBRegressor()
        xgb_reg.load_model('_models/'+station+'/xgb.model')
        # print(station, 'loaded (xgb)')
        
        scaler_inp = joblib.load('_models/'+station+'/scaler99.pkl')
        
        # data
        c_inp = pd.read_csv('data/level2/'+station+'/'+station+'_inp_merged.csv',
                             index_col=0)
        c_exp = pd.read_csv('data/level2/'+station+'/'+station+'_exp_merged.csv',
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



# ____________________________________________________________________

# plot NSE bar plot of XGB models
#
def presentation_plot_fit_xgb_retrain():
    import xgboost as xgb
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    
    big = []
    for station in os.listdir('_models'):
        if station == '39046':
            continue
        
        # data
        c_inp = pd.read_csv('data/level2/'+station+'/'+station+'_inp_merged.csv',
                             index_col=0)
        c_exp = pd.read_csv('data/level2/'+station+'/'+station+'_exp_merged.csv',
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
















