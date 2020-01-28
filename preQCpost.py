import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


path = 'data/level3/'


# timeseries
#
for i in os.listdir(path):
    # print(i)
    dt = pd.read_csv(path+i+'/comp/'+i+'_merged.csv', index_col=0)
    
    idx = dt[dt[i] != dt['orig']]
    
    if not idx.empty:
        idx = idx.index[0]
    
        dt = dt[idx:]
        
        plt.figure(figsize=(10, 4), dpi=300)
        dt['orig'].plot(c='darkcyan', label='preQC')
        dt[i].plot(c='black', label='QCd')
        plt.legend()
        plt.yscale("log")
        plt.savefig('QC_plots/'+i+'.png')
        plt.close()
     
        
# map   
#
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
