import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


''' ___________________________ HEADER PARS ______________________________ '''

# dir path
path = 'data/level3/'

# meta for station names
meta_id_name = pd.read_csv('meta/nrfa_station_info.csv')[['id', 'name']]

# ids of stations with significant QC corrections (subset)
IDS = [35003, 39096, 39084, 47019, 39125, 38014, 39056, 30014, 33055, 39049,
       40017, 33018, 33023, 41027, 37015, 38007, 41030, 33031, 39058, 33030,
       49006, 68020, 34010, 28015, 34018, 30002, 46005, 38022, 38021, 28044,
       54017, 34019, 33035, 30012, 41016, 32031]


''' ____________________________ TIMESERIES ______________________________ '''

# for i in os.listdir(path):

for station_id in IDS:
    print(station_id)
    i = str(station_id)
    
    if i == '49006':
        continue
    
    dt = pd.read_csv(path+i+'/'+i+'_qc.csv', index_col=0)
    dt_sub = dt[dt[i] != dt['orig']]
    print(dt_sub)
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
        plt.savefig('QC_plots/'+i+'.png')
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
