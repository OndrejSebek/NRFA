import pandas as pd
import numpy as np

import os

import QC_utils as qc_u


""" ___________________ nr. of inps with qc changes _____________________ """

IDS = [33013, 34010, 34012, 34018, 39056, 40017, 46005, 47019, 48001, 49006]

qc_corr = pd.read_csv('meta/_NRFA_qc/gdf-live-audit-counts-2020-02-17.csv',
                      index_col=1)["STATION"].unique()

big = []
for st in IDS:
    x = pd.read_csv(f"_model_inps/xgbsearch_{st}.csv")
    inps = x["var"].str[:5].unique()
    
    c = 0
    b = []
    for inp in inps:
        if int(inp) in qc_corr:
            c += 1
            b.append(inp)
    big.append([st, c]+b)

big = pd.DataFrame(big)


""" _______________________ preqcF x preqcT ______________________________ """

from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

IDS = [33013, 34010, 34012, 34018, 39056, 40017, 46005, 47019, 48001, 49006]

big = []
for st in IDS:
    t = pd.read_csv(f"sens/preqcT/{st}/{st}_merged.csv",
                    index_col=0)
    f = pd.read_csv(f"sens/preqcF/{st}/{st}_merged.csv",
                    index_col=0)
    
    locs = t[t["nn_m"] != f["nn_m"]].index
    
    t = t.loc[locs]
    f = f.loc[locs]
    
    t_rmse = sqrt(mean_squared_error(t[str(st)], t["nn_m"]))
    f_rmse = sqrt(mean_squared_error(f[str(st)], f["nn_m"]))
    
    big.append([st, locs.shape[0], f_rmse, t_rmse])
    
    pd.merge(f[[str(st), "nn_m"]], t["nn_m"],
             left_index=True, right_index=True,
             suffixes=["_postqc", "_preqc"]).plot(figsize=(15,8),
                                                  color=["black", "firebrick", "darkcyan"])
    plt.savefig(f"sens/plots/{st}.png",
                dpi=300)

big = pd.DataFrame(big, columns=["st_id", "n_diffs", "preqcF", "preqcT"])
big["preqcT"]/big["preqcF"]


# check preqcT x preqcF inps











