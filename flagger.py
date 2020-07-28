from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

IDS = [33013, 34010, 34012, 34018, 39056, 40017, 46005, 47019, 48001, 49006]

big = []
for st in IDS:
    merged = pd.read_csv(f"data/level3/{st}/{st}_merged.csv",
                 index_col=0)
    # merged = merged.loc["2015-09-01":]
    
    avgQ = merged[str(st)].mean()
    
    f1 = 0
    for std in range(1, 20):
        for abs_d in np.arange(0, avgQ, avgQ/20):
            fl = ((abs(merged['nn_m'] - merged['orig']) > std*merged['nn_std'])
                  & (abs(merged['nn_m'] - merged['orig']) > abs_d))
            
            fl_qc = ~(merged[merged.columns[0]] == merged['orig']) #.values
            
            f1_c = f1_score(fl, fl_qc)
            
            if f1_c > f1:
                f1 = f1_c
                std_f1 = std
                abs_d_f1 = abs_d
            
    big.append([st, std_f1, abs_d_f1, f1])
    
x = pd.DataFrame(big,
                 columns=["st_id", "std", "abs_d", "f1_max"])

x.to_csv(f"data/def_flaggers/def_flaggers.csv", index=False)
