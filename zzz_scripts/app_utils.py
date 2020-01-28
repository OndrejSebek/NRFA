from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import KNeighborsClassifier


def flag_outliers_fixed_abs(merged, abs_d):
    fl = (abs(merged['nn_m'] - merged['orig'])).astype(float) > abs_d
    return fl


def flag_outliers_zscore(merged, std, abs_d):
    fl = (abs(merged['nn_m'] - merged['orig']) > std*abs(merged['nn_std'])) & (abs(merged['nn_m'] - merged['orig']) > abs_d)
    return fl

def flag_outliers_kde(merged, nn_preds, smoothing):
    kdes = []
    for i in range(merged.shape[0]):
        kdes.append(gaussian_kde(nn_preds.iloc[i], bw_method=(smoothing/nn_preds.values.std(ddof=1))))

    flags = []
    for i, val in enumerate(merged['orig']):
        if kdes[i].evaluate(val) == 0:
            flags.append(True)
        else:
            flags.append(False)

    # flag three consecutive values outside estimated PDE
    flags_tr = []
    for i, val in enumerate(flags):
        if i < len(flags)-1: 
            if all([val, flags[i-1], flags[i+1]]):
                flags_tr.append(True)
            else:
                flags_tr.append(False)
        else: 
            flags_tr.append(False)

    return flags, flags_tr


# def flag_outliers_iforest(merged, nn_preds):
#     nn_preds.index = merged.index
#     if_data = pd.merge(nn_preds, merged['orig'], left_index=True, right_index=True)
#     clf = IsolationForest(max_samples=6, contamination=.1, behaviour="new")
    
#     flags = []
#     for i in if_data.index:
#         if_dt_ = if_data.loc[i].values.reshape(-1, 1)
#         scores = clf.fit_predict(if_dt_)
        
#         if all(scores == [1, 1, 1, 1, 1, -1]):
#             flags.append(True)
#         else:
#             flags.append(False)
    
#     return flags
      

# def flag_outliers_knn(merged, nn_preds):
#     nn_preds.index = merged.index
#     if_data = pd.merge(nn_preds, merged['orig'], left_index=True, right_index=True)

#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(X_train, y_train)
    

def flag_qc_corrections(merged):
    return ~(merged[merged.columns[0]] == merged['orig']).values
    
    
def fig_comb(merged, Z_s, flags, flags_qc, rel_errors, log_opt):
    st_id = merged.columns[0]
    
    nn_h = merged['nn_m']+Z_s*merged['nn_std']
    nn_l = merged['nn_m']-Z_s*merged['nn_std']
    
    if len(flags) == 0:
        flags_x = []
        flags_vals = []
    else:
        flags = np.array(flags)
        flags_x = merged.index[flags]
        if log_opt:
            flags_vals = [(merged[st_id].max()+.1*(merged[st_id].max()-merged[st_id].min()))]*len(flags_x)
        else:
            flags_vals = [(merged[st_id].min()-.1*(merged[st_id].max()-merged[st_id].min()))]*len(flags_x)
        
    flags_qc_x = merged.index[flags_qc]
    if log_opt:
        flags_qc_vals = [(merged[st_id].max()+.2*(merged[st_id].max()-merged[st_id].min()))]*len(flags_qc_x)
    else:
        flags_qc_vals = [(merged[st_id].min()-.2*(merged[st_id].max()-merged[st_id].min()))]*len(flags_qc_x)
    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, vertical_spacing=0.02)
    
    # update plot background color to transparent
    fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)',
                         margin_l=0, margin_t=0)
    
    
    ''' add traces top '''
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged['orig'], 
                             name="preqc",
                             legendgroup='preqc',
                             line=dict(color='darkcyan', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged['nn_m'], 
                             name="nn",
                             legendgroup='nn',
                             line=dict(color='firebrick', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged[str(st_id)], 
                             name="qcd",
                             legendgroup='qcd',
                             line=dict(color='black', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=flags_x,
                             y=flags_vals, 
                             name="flags",
                             mode='markers',
                             marker_color='crimson',
                             hoverinfo="x",
                             marker_line_width=.5,
                             marker_size=6,
                             marker_symbol='triangle-up'
                             ),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=flags_qc_x,
                             y=flags_qc_vals, 
                             name="qc flags",
                             mode='markers',
                             marker_color='lightgrey',
                             hoverinfo="x",
                             marker_line_width=.5,
                             marker_size=6,
                             marker_symbol='triangle-up'
                             ),
                  row=1, col=1)
    
    
    # errors/resids
    nn_l_errors = nn_l-merged['orig']
    nn_h_errors = nn_h-merged['orig']
    orig_errors = merged['orig']-merged['orig']
    nn_errors = merged['nn_m']-merged['orig']
    qcd_errors = merged[st_id]-merged['orig']

    if rel_errors:
        nn_l_errors = nn_l_errors/merged['orig']
        nn_h_errors = nn_h_errors/merged['orig']
        orig_errors = orig_errors/merged['orig']
        nn_errors = nn_errors/merged['orig'] 
        qcd_errors = qcd_errors/merged['orig']
    
    
    ''' add traces bot '''
    fig.add_trace(go.Scatter(x=merged.index,
                             y=nn_h_errors, 
                             name="nn_h",
                             legendgroup='nn',
                             showlegend=False,
                             line=dict(color='lightcoral', width=2)),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=nn_l_errors, 
                             fill='tonexty',
                             fillcolor='lightcoral',
                             name="nn_l", 
                             legendgroup='nn',
                             showlegend=False,
                             line=dict(color='lightcoral', width=2)),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=orig_errors, 
                             name="preqc", 
                             legendgroup='preqc',
                             showlegend=False,
                             line=dict(color='darkcyan', width=2)),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=nn_errors, 
                             name="nn", 
                             legendgroup='nn',
                             showlegend=False,
                             line=dict(color='firebrick', width=2)),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=merged[st_id].index,
                             y=qcd_errors, 
                             name="qcd",
                             legendgroup='qcd',
                             showlegend=False,
                             line=dict(color='black', width=2)),
                  row=2, col=1)
    
    if log_opt:
        fig.update_layout(yaxis_type="log")
    return fig




def fig_comb_nns(merged, nn_preds, flags, flags_qc, rel_errors, log_opt):
    st_id = merged.columns[0]
    
    if len(flags) == 0:
        flags_x = []
        flags_vals = []
    else:
        flags = np.array(flags)
        flags_x = merged.index[flags]
        if log_opt:
            flags_vals = [(merged[st_id].max()+.1*(merged[st_id].max()-merged[st_id].min()))]*len(flags_x)
        else:
            flags_vals = [(merged[st_id].min()-.1*(merged[st_id].max()-merged[st_id].min()))]*len(flags_x)

    
    flags_qc_x = merged.index[flags_qc]
    if log_opt:
        flags_qc_vals = [(merged[st_id].max()+.2*(merged[st_id].max()-merged[st_id].min()))]*len(flags_qc_x)
    else:
        flags_qc_vals = [(merged[st_id].min()-.2*(merged[st_id].max()-merged[st_id].min()))]*len(flags_qc_x)

    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, vertical_spacing=0.02)
    
    # update plot background color to transparent
    fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)',
                         margin_l=0, margin_t=0)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged['orig'], 
                             name="preqc",
                             legendgroup='preqc',
                             line=dict(color='darkcyan', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged['nn_m'], 
                             name="nn",
                             legendgroup='nn',
                             line=dict(color='firebrick', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=merged[str(st_id)], 
                             name="qcd",
                             legendgroup='qcd',
                             line=dict(color='black', width=2)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=flags_x,
                             y=flags_vals, 
                             name="flags",
                             mode='markers',
                             marker_color='crimson',
                             hoverinfo="x",
                             marker_line_width=.5,
                             marker_size=6,
                             marker_symbol='triangle-up'
                             ),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=flags_qc_x,
                             y=flags_qc_vals, 
                             name="qc flags",
                             mode='markers',
                             marker_color='lightgrey',
                             hoverinfo="x",
                             marker_line_width=.5,
                             marker_size=6,
                             marker_symbol='triangle-up'
                             ),
                  row=1, col=1)
    
    
    # errors/resids
    orig_errors = merged['orig']-merged['orig']
    qcd_errors = merged[st_id]-merged['orig']
    nn_errors = merged['nn_m']-merged['orig']
    
    if rel_errors:
        orig_errors = orig_errors/merged['orig']
        qcd_errors = qcd_errors/merged['orig']
        nn_errors = nn_errors/merged['orig'] 
        
    
    fig.add_trace(go.Scatter(x=merged.index,
                             y=orig_errors, 
                             name="preqc", 
                             legendgroup='preqc',
                             showlegend=False,
                             line=dict(color='darkcyan', width=2)),
                  row=2, col=1)
    
    for i in nn_preds:
        _nn_errors = (nn_preds[i].values-merged['orig'].values)
        if rel_errors:
            _nn_errors = _nn_errors/merged['orig'].values
                
        fig.add_trace(go.Scatter(x=merged.index,
                                 y=_nn_errors, 
                                 name="nn_"+str(i), 
                                 legendgroup='nn',
                                 showlegend=False,
                                 line=dict(color='firebrick', width=.5, dash='dash')), #firebrick
                      row=2, col=1)
        
    fig.add_trace(go.Scatter(x=merged.index,
                             y=nn_errors, 
                             name="nn", 
                             legendgroup='nn',
                             showlegend=False,
                             line=dict(color='firebrick', width=2)),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=merged[st_id].index,
                             y=qcd_errors, 
                             name="qcd",
                             legendgroup='qcd',
                             showlegend=False,
                             line=dict(color='black', width=2)),
                  row=2, col=1)
    
    if log_opt:
        fig.update_layout(yaxis_type="log")
    return fig    
    
    
def heatmap():
    dt = pd.read_csv('meta/comp/Qn_stats.csv', index_col=0)
    
    rmses = dt[dt.columns[:4]].sort_index(ascending=False)
    stds = dt[dt.columns[4:-1]].sort_index(ascending=False)
    
    x_labs = ['all', '<Q70', 'Q30-Q70', '>Q30']
    y_labs = rmses.index.astype(str).values+':'
    
    fig = make_subplots(rows=1, cols=2, 
                        shared_yaxes=True, shared_xaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02,
                        subplot_titles=['nRMSE', 'nSTD'])
    
    # update plot background color to transparent
    fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)',
                         margin_l=0)
    
    fig.add_trace(go.Heatmap(x=x_labs,
                             y=y_labs,
                             z=rmses,
                             colorscale='Tealrose',
                             name='nRMSE'),
                  row=1, col=1)

    fig.add_trace(go.Heatmap(x=x_labs,
                             y=y_labs,
                             z=stds,
                             colorscale='Tealrose',
                             name='nSTD'),                  
                  row=1, col=2)
    
    # fig['data'][0]['showscale']=False
    fig['data'][1]['showscale']=False
    
    # export offline
    # pio.write_html(fig, file='plots/Qn_rmse_std_heatmap.html', auto_open=False)
    
    return fig
    
def heatmap_model_inps(station):
    dt = pd.read_csv('_model_inps_subtable/'+str(station)+'.csv',
                     index_col=0)
    
    fig = go.Figure(go.Heatmap(x='t-'+dt.columns.astype(str),
                               y=dt.index.astype(str).values+':',
                               z=dt.values,
                               zmin=0, zmax=2,
                               colorscale='PuBu',
                               name='st inps'))
    
    fig['data'][0]['showscale'] = False
    fig['layout']['xaxis'].update(side='top')
    
    return fig



