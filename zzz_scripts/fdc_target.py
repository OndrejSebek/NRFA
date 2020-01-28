import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


station_id = '46014'
inps = pd.read_csv('data/inps/'+station_id+'/'+station_id+'_inp_merged.csv',
                               index_col=0, header=0)
exp = pd.read_csv('data/inps/'+station_id+'/'+station_id+'_exp_merged.csv',
                               index_col=0, header=0)

exp_perc = exp.sort_values('46014', ascending=False)
percs = pd.DataFrame(np.arange(1, exp.shape[0]+1)/(exp.shape[0]+1) * 100, columns=['perc'])
percs.index = exp_perc.index

valperc = pd.concat([exp_perc, percs], axis=1)



x = pd.merge(exp, valperc, left_index=True, right_index=True)



perc_out_exp = x[['perc']]
perc_out_exp.to_csv('data/inps/'+station_id+'/'+station_id+'_exp_perc.csv')


scaler_inp = preprocessing.StandardScaler()
inps = scaler_inp.fit_transform(inps)


x_cal, x_val, y_cal, y_val = train_test_split(inps, perc_out_exp.values, 
                                              test_size=0.3, 
                                              random_state=None,
                                              shuffle=True)

cal_dataset = tf.data.Dataset.from_tensor_slices((x_cal, y_cal))
cal_dataset = cal_dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32)



model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_val.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')])

model.compile(optimizer=tf.keras.optimizers.Adam(.0001),
              loss=['mse'],
              metrics=['RootMeanSquaredError'])  

# early stopping callback
cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                              patience=20, verbose=0, mode='auto',
                                              baseline=None, restore_best_weights=True)

# reduce LR callback
cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                   patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0,
                                                   min_lr=0)


model.fit(cal_dataset, epochs=10000,
          validation_data=val_dataset, 
          callbacks=[cb_es, cb_rlr],
          verbose=2)



preds = model.predict(inps)

preds = pd.DataFrame(preds, columns=['pred'])
preds.index = perc_out_exp.index




plt.figure(figsize=(20,8), dpi=300)
plt.plot(preds.values)
plt.plot(perc_out_exp.values)
plt.savefig('plots/'+station_id+'/perc_test.jpg')






















