import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = pd.read_csv('out_RMSEs/keras_RMSE_opts_600.csv', index_col=0).reset_index(drop=True)

plt.figure()
plt.plot(x['range'], x['val'], 'x')

x = pd.read_csv('keras_RMSE.csv', index_col=0).reset_index(drop=True)


print(np.corrcoef(x['range'], x['val'])[1, 0])




def boxplots(opt, x):
    opts = np.unique(x[opt])
    
    for ep in ['cal', 'val']:       
        dt = pd.DataFrame()
        for i in opts:
            sub = x[x[opt] == i][ep]
            sub.name = i
            sub.index = range(sub.shape[0])
        
            dt = pd.concat([dt, sub], axis=1)

        plt.figure(figsize=(10,8), dpi=200)
        dt.boxplot(grid=False, rot=45)
        plt.savefig('post/'+opt+'_'+ep+'.jpg')
        plt.close()


colnames = list(map(str, x.columns))
colnames.remove('cal')
colnames.remove('val')

for opt in colnames:
    boxplots(opt, x)
    
    
    
    
def plots(opt, x):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6), dpi=200)
    
    ax1.plot(x[opt], x['cal'], 'x')
    ax1.set_title('cal')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel(opt)
    ax2.plot(x[opt], x['val'], 'x')
    ax2.set_title('val')
    
    fig.savefig('post_plots/'+opt+'.jpg')
    plt.close()


for opt in x.columns:
    plots(opt, x)

    
    

best = x.sort_values('val')
best.to_csv('out_pres/RMSE_table.csv', index=False)
best = best.iloc[:10]
