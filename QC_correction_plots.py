import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


''' ___________________________ HEADER PARS ______________________________ '''

# dir path
path = 'data/level3/'

# meta for station names
meta_id_name = pd.read_csv('meta/nrfa_station_info.csv')[['id', 'name', 'measuring-authority-station-id']]

# ids of stations with significant QC corrections (subset)
IDS = [35003, 39096, 39084, 47019, 39125, 38014, 39056, 30014, 33055, 39049,
       40017, 33018, 33023, 41027, 37015, 38007, 41030, 33031, 39058, 33030,
       49006, 68020, 34010, 28015, 34018, 30002, 46005, 38022, 38021, 28044,
       54017, 34019, 33035, 30012, 41016, 32031]


''' ____________________________ TIMESERIES ______________________________ '''

# for i in os.listdir(path):

big = []
for station_id in IDS:
    print(station_id)
    i = str(station_id)
    
    if i == '49006':
        continue
    
    dt = pd.read_csv(path+i+'/'+i+'_qc.csv', index_col=0)
    # dt_sub_idx = dt[dt[i] != dt['orig']].index[0]
    # dt_sub = dt.loc[dt_sub_idx:]
    dt_sub = dt[dt[i] != dt['orig']]
    # print(dt_sub)
    
    # periods
    thr = 7
    pers = pd.DataFrame([dt_sub.index[:-1], dt_sub.index[1:]]).T
    pers['step'] = (pd.to_datetime(pers[1]) - pd.to_datetime(pers[0])).dt.days
    steps = pers[pers['step'] > thr]
    steps.columns = ['l', 'r', 'step']
    
    cur = [i, meta_id_name[meta_id_name['id'] == int(i)]['measuring-authority-station-id'].values[0]]
    start_day = pers.loc[pers.index[0], pers.columns[0]]
    for idx in steps.index:
        end_day = steps.loc[idx, 'l']
        cur.extend([str(start_day)+' - '+str(end_day), 'comment'])
        start_day = steps.loc[idx, 'r']
    end_day = pers.loc[pers.index[-1], pers.columns[1]]
    cur.extend([str(start_day)+' - '+str(end_day), 'comment'])
    
    big.append(cur)
        
    if not dt_sub.empty:
        # find name
        name = str(i)+': '+meta_id_name[meta_id_name.id == int(i)].name.values[0]
        idx = dt_sub.index[0]
        
        # add 2 extra months (lazy)
        if idx[6] == '0':
            idx = idx[:5]+'08'+idx[7:]
        else:
            idx = idx[:6]+str(int(idx[6])-2)+idx[7:]

        # data subset with QC corrs
        dt_plot = dt[idx:]
        dt_plot.index = pd.to_datetime(dt_plot.index)
        dt_sub.index = pd.to_datetime(dt_sub.index)
        
        # find qc corr periods
        start_flags = []
        end_flags = []
        for q in range(2, len(cur), 2):
            start_flags.append(cur[q][:10])
            end_flags.append(cur[q][13:])
                
        # QC correction plot
        fig = plt.figure(figsize=(10, 4), dpi=300)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        
        # timeseries
        ax0.plot(dt_plot['orig'], c='darkcyan', label='preQC')
        ax0.plot(dt_plot[i], c='black', label='QCd')
        ax0.title.set_text(name)
        ax0.set_yscale('log')
        ax0.legend()
        
        # flags + periods
        ax1.plot(dt_plot[i].index, [.5]*len(dt_plot[i].index),
                 c='white', marker="|", linestyle='', label='')
        ax1.plot(dt_sub.index, [.5]*len(dt_sub.index),
                 c='darkcyan', marker="|", linestyle='', label='')
        ax1.plot(pd.to_datetime(start_flags), [1]*len(start_flags), 
                 c='black', marker='>', linestyle='')
        ax1.plot(pd.to_datetime(end_flags), [1]*len(start_flags),
                 c='black', marker='<', linestyle='')
        ax1.set_ylim([0, 1.5])
        ax1.set_yticks([])
        
        # save plots
        plt.savefig('QC_plots/'+i+'.png')
        plt.close()

# format + export csvs
big = pd.DataFrame(big)
big.columns = big.columns.astype(str)
big.columns = ['NRFA_ID', 'EA_ID']+list(map(str, range(big.shape[1]-2)))
big.sort_values('NRFA_ID', inplace=True)

big.to_csv('QC_plots/csv/NRFA_QC_corr_class.csv', index=False)
big.to_excel('QC_plots/csv/NRFA_QC_corr_class.xlsx', index=False)


''' ______________________________ MAP ___________________________________ '''

nrfa_meta = pd.read_csv('meta/NRFA_meta.csv')
stations = pd.DataFrame(os.listdir(path), columns=['id'], dtype=int)
stations = pd.merge(stations, nrfa_meta, 
                    left_on='id', right_on='NRFA_ID',
                    how='inner')

fig = plt.figure(figsize=(10, 10), dpi=300)
plt.plot(stations['easting'], stations['northing'],
         c='black',
         marker = 'o', linestyle=' ')
plt.savefig('QC_plots/map.png')
plt.close()


# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# # ax.stock_img()
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.COASTLINE)
# ax.set_extent([-5, 30, 40, 65], crs=ccrs.PlateCarree())

# plt.plot(stations['easting'], stations['northing'], c='black')
# plt.savefig('QC_plots/map.png')
# plt.close()
