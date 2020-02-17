import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


path = '../data/level3/'

meta_id_name = pd.read_csv('../meta/nrfa_station_info.csv')[['id', 'name']]


''' ____________________________ TIMESERIES ______________________________ '''

for i in os.listdir(path):
    print(i)
    if i == '49006':
        continue
    
    dt = pd.read_csv(path+i+'/comp/'+i+'_merged.csv', index_col=0)
    dt_sub = dt[dt[i] != dt['orig']]
    
    if not dt_sub.empty:
        # find name
        name = str(i)+': '+meta_id_name[meta_id_name.id == int(i)].name.values[0]
        idx = dt_sub.index[0]
        
        # add 2 extra months (lazy)
        if idx[6] == '0':
            idx = idx[:5]+'08'+idx[7:]
        else:
            idx = idx[:6]+str(int(idx[6])-2)+idx[7:]

        # subset data
        dt_plot = dt[idx:]
        
        # plot
        plt.figure(figsize=(10, 4), dpi=300)
        dt_plot['orig'].plot(c='darkcyan', label='preQC')
        dt_plot[i].plot(c='black', label='QCd')
        plt.legend()
        plt.title(name)
        plt.yscale("log")
        plt.savefig('../QC_plots/'+i+'.png')
        plt.close()


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
