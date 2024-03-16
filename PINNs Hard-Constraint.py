#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:38:48 2023

@author: dennisirwanto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:07:16 2023

@author: dennisirwanto
"""

'''
ALGORITMA TRIAL FORM SOLUTION UNTUK PDP BLACK-SCHOLES
'''

'''
Kalau ERROR "tensor object cannot be converted to numpy array", maka restart python
dengan update "!pip install --upgrade tensorflow" di terminal.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
import tensorflow as tf
import scipy.stats as stats
import random

def A(S,t, S_max):    # Mendefiniskan A(x,y)
    K = 100
    return (t) * np.maximum(S - (K / S_max), 0) + (1-t) * S * (1 - (K / S_max))

# Pembentukkan PDP
def pde_system(S, t, S_max, T, sigma, r, net):
    S=S.reshape(-1,1)
    t=t.reshape(-1,1)
    S=tf.constant(S,dtype=tf.float32)
    t=tf.constant(t,dtype=tf.float32)
    
    T=tf.constant(T,dtype=tf.float32)
    S_max = tf.constant(S_max,dtype=tf.float32)
 
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(S)
        tape.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(S)
            tape1.watch(t)
            merge_input=np.stack([S,t],axis=1)              # Variabel bebas x & y dimerge (digabungkan) agar dapat dijalankan dalam neural networks secara berbarengan
            f=A(S,t, S_max)+(1-t)*S*(1-S)*net(merge_input)     # Trial Neural Form
        f_S=tape1.gradient(f,S)                         # Turunan pertama trial neural form thd x
        f_t=tape1.gradient(f,t)                         # Turunan pertama trial neural form thd y
    f_SS=tape.gradient(f_S,S)                           # Turunan kedua trial neural form thd x
    f_tt=tape.gradient(f_t,t)                           # Turunan kedua trial neural form thd y
    
    
    pde_loss=(f_t) + (1/2) * (sigma**2) * S**2 * f_SS + r * S * f_S - r * f 
    
    square_loss=tf.square(pde_loss)                         # Mengkuadratkan fungsi loss
    total_loss=tf.reduce_mean(square_loss)                  # Mencari nilai mean dari loss
    return total_loss

# train
N_S = 100
N_t = 100

train_S=np.linspace(0, 1, 1000).astype(np.float32)                              # Membuat 100 test x, dengan range yg sama yaitu x- [0,2]
train_t=np.linspace(0,1, 1000).astype(np.float32)

train_S_accent=np.linspace(0.00, 200, N_S).astype(np.float32)                              # Membuat 100 test x, dengan range yg sama yaitu x- [0,2]

max_train_S_accent = np.max(train_S_accent)

max_train_T = np.max(train_t)  


K = 100
sigma = 0.2
r = 0.05

# define network -- ARSITEKTUR LAYER NEURAL NETWORK
NN=tf.keras.models.Sequential([
    tf.keras.layers.Input((2,)),                            # Input Size = 2 (karena ada 2 variabel bebas)
    tf.keras.layers.Dense(units=16,activation='sigmoid'),   # Hidden Layer Size = 32 (bebas sesuai kebutuhan)
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    #tf.keras.layers.Dense(units=16,activation='sigmoid'),
    #tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='linear')                          # Output Size = 1 (Output langsung nilai t)
    ])
NN.summary()
optm=tf.keras.optimizers.legacy.Adam() 
#optm=tf.keras.optimizers.Adam(learning_rate=0.00001) 

train_loss_record=[]

for itr in range(10000):                                    # Melatih neural network sebanyak 30.000 kali iterasi
    with tf.GradientTape() as tape:
        train_loss=pde_system(train_S,train_t,max_train_S_accent, max_train_T, sigma, r, NN)           # Melatih input training X dan Y
        train_loss_record.append(train_loss)                # Menggabungkan loss-loss untuk kebutuhan grafik loss
        grad_w=tape.gradient(train_loss,NN.trainable_variables)     # Mencari gradien loss
        optm.apply_gradients(zip(grad_w,NN.trainable_variables))    # Mengoptimasi bobot dan bias untuk iterasi berikutnya
        
    if itr%1000==0:                                         # Print loss setiap 1000 iterasi
        print(itr, train_loss.numpy())

# Plot Grafik perkembangan minimalisir loss
plt.figure(figsize=(10,8))
plt.plot(train_loss_record, color = 'orange', label='Train Loss')
plt.xlabel('Epochs', fontsize=16)  # Adding label for X-axis with increased font size
plt.ylabel('Loss', fontsize=16)  # Adding label for Y-axis with increased font size
plt.legend(fontsize=18)  # Displaying the legend with increased font size
plt.yscale('log')  # Setting the y-axis scale to logarithmic
plt.ylim(1e-6, None)  # Setting the y-axis limits from 10^-6 to the maximum value

plt.show()



# test - Membuat Variabel test/uji untuk menguji neural network yg telah dibuat
max_test_T = np.full_like(train_t, max_train_T)

test_S1= np.linspace(0, 1, N_S).astype(np.float32)                              # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
test_S2= np.linspace(0, 200, N_S).astype(np.float32)   

test_t= np.linspace(0, 1, N_t).astype(np.float32) 

#test_S= train_S * max_train_S_accent                                 # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
#test_t= max_test_T - train_t * max_test_T

max_test_T = np.max(test_t) # waktu jatuh tempo T = 1
max_test_S2 = np.max(test_S2)
                                
testS1,testt1=np.meshgrid(test_S1,test_t)                       # Menggabungkan x dan y dalam satu grid
testSa=testS1.reshape(-1)

testS2,testt2 = np.meshgrid(test_S2, test_t)
testSb=testS2.reshape(-1)

testta=testt1.reshape(-1)
testtb=testt2.reshape(-1)

Appred=A(testSa,testta,max_test_S2)
merge_input=np.stack([testSa,testta],axis=1)
pred_f = Appred + testSa * (1-testta) * (1 - testSa) * NN.predict(merge_input).ravel()
pred_f = pred_f * max_test_S2
predf = pred_f.reshape((N_S, N_t))

# Mendefinisikan solusi analitik
def solusi_analitik_call(S, K, T, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    f = S * N_d1 - K * np.exp(-r * (T-t)) * N_d2
    return f

true_f = solusi_analitik_call(testSb, K, max_test_T, testtb, r, sigma)
#true_f = true_f / max_test_S2
true_f = true_f.reshape((100,100))

err=predf-true_f

fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
surf=ax.plot_surface(testS2,testt2,predf,linewidth=0,antialiased=False, cmap = 'viridis')
surf=ax.set_xlabel('Harga Aset')
surf=ax.set_ylabel('waktu')
surf=ax.set_zlabel('Harga Opsi')
surf=ax.view_init(elev=20, azim=240)

fig2,ax2=plt.subplots(subplot_kw={"projection":"3d"})
surf2=ax2.plot_surface(testS2,testt2,true_f,linewidth=0,antialiased=False, cmap = 'viridis')
surf2=ax2.set_xlabel('Harga Aset')
surf2=ax2.set_ylabel('waktu')
surf2=ax2.set_zlabel('Harga Opsi')
surf2=ax2.view_init(elev=20, azim=240)

fig3,ax3=plt.subplots(subplot_kw={"projection":"3d"})
surf3=ax3.plot_surface(testS2,testt2,err,linewidth=0,antialiased=False, cmap = 'viridis')
surf3=ax3.set_xlabel('Harga Aset')
surf3=ax3.set_ylabel('waktu')
surf3=ax3.set_zlabel('Harga Opsi')
surf3=ax3.view_init(elev=20, azim=240)

fig4,ax4=plt.subplots()
surf4=ax4.pcolormesh(testS2,testt2,err,cmap = 'viridis')
fig4.colorbar(surf4, label='Penyebaran Error')
surf4=ax4.set_xlabel('Harga Aset')
surf4=ax4.set_ylabel('waktu')
#surf3=ax3.view_init(elev=20, azim=240)


########## SLICE TEST PADA t=0
test_t3= 0

#test_S= train_S * max_train_S_accent                                 # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
#test_t= max_test_T - train_t * max_test_T

max_test_T = np.max(test_t) # waktu jatuh tempo T = 1
max_test_S2 = np.max(test_S2)
                                
testS3,testt3=np.meshgrid(test_S1,test_t3)                       # Menggabungkan x dan y dalam satu grid
testSc=testS3.reshape(-1)

testS4,testt3 = np.meshgrid(test_S2, test_t3)
testSd=testS4.reshape(-1)

testtc=testt3.reshape(-1)
testtd=testt3.reshape(-1)

Appred_t0=A(testSc,testtc,max_test_S2)
merge_input_t0=np.stack([testSc,testtc],axis=1)
pred_f_t0 = Appred_t0 + testSc * (1-testtc) * (1 - testSc) * NN.predict(merge_input_t0).ravel()
pred_f_t0 = pred_f_t0 * max_test_S2
predf_t0 = pred_f_t0.reshape((N_S, 1))

true_f_t0 = solusi_analitik_call(testSd, K, max_test_T, testtd, r, sigma)
#true_f_t0 = true_f_t0 / max_test_S2
true_f_t0 = true_f_t0.reshape((N_S,1))

# Plot the first line
# Plot the first line
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(testSd, predf_t0, label='Prediksi Neural Network', color='tomato', linestyle='--')
ax5.plot(testSd, true_f_t0, label='Solusi Analitik BS', color='green', linestyle='-')
ax5.set_xlabel('Harga Aset')
ax5.set_ylabel('Harga Opsi')
ax5.set_title('Perbandingan Hasil Harga Opsi pada t=0')
ax5.legend()

# Calculate RMSE
RMSE = np.sqrt(np.mean(np.sum((true_f_t0-predf_t0)**2)))
print('RMSE =',RMSE)

# Calculate MAE
MAE =np.mean(np.sum(np.abs(true_f_t0-predf_t0)))
print('MAE =', MAE)

# Calculate R-squared
mean_true_values = np.mean(true_f_t0)

numerator = np.sum((true_f_t0 - predf_t0) ** 2)
denominator = np.sum((true_f_t0 - mean_true_values) ** 2)
r_squared = 1 - (numerator / denominator)
print('r_squared =', r_squared)

### HASIL NORMALISASI ( BELUM DI REVERT KEMBALI )###
# max_test_T = np.full_like(train_t, max_train_T)

# test_S1= np.linspace(0, 1, N_S).astype(np.float32)                              # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
# test_S2= np.linspace(0, 250, N_S).astype(np.float32)   

# test_t= np.linspace(0, 1, N_t).astype(np.float32) 

# #test_S= train_S * max_train_S_accent                                 # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
# #test_t= max_test_T - train_t * max_test_T

# max_test_T = np.max(test_t) # waktu jatuh tempo T = 1
# max_test_S2 = np.max(test_S2)
                                
# testS1,testt1=np.meshgrid(test_S1,test_t)                       # Menggabungkan x dan y dalam satu grid
# testSa=testS1.reshape(-1)

# testS2,testt2 = np.meshgrid(test_S2, test_t)
# testSb=testS2.reshape(-1)

# testta=testt1.reshape(-1)
# testtb=testt2.reshape(-1)

# Appred=A(testSa,testta,max_test_S2)
# merge_input=np.stack([testSa,testta],axis=1)
# pred_f = Appred + testSa * (1-testta) * (1 - testSa) * NN.predict(merge_input).ravel()
# #pred_f = pred_f * max_test_S2
# predf = pred_f.reshape((N_S, N_t))

# # Mendefinisikan solusi analitik
# def solusi_analitik_call(S, K, T, t, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
#     d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
#     N_d1 = stats.norm.cdf(d1)
#     N_d2 = stats.norm.cdf(d2)
#     f = S * N_d1 - K * np.exp(-r * (T-t)) * N_d2
#     return f

# true_f = solusi_analitik_call(testSb, K, max_test_T, testtb, r, sigma)
# true_f = true_f / max_test_S2
# true_f = true_f.reshape((100,100))

# err=predf-true_f

# fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
# surf=ax.plot_surface(testS1,testt1,predf,linewidth=0,antialiased=False, cmap = 'viridis')
# surf=ax.set_xlabel('Harga Aset')
# surf=ax.set_ylabel('waktu')
# surf=ax.set_zlabel('Harga Opsi')
# surf=ax.view_init(elev=20, azim=240)

# fig2,ax2=plt.subplots(subplot_kw={"projection":"3d"})
# surf2=ax2.plot_surface(testS1,testt1,true_f,linewidth=0,antialiased=False, cmap = 'viridis')
# surf2=ax2.set_xlabel('Harga Aset')
# surf2=ax2.set_ylabel('waktu')
# surf2=ax2.set_zlabel('Harga Opsi')
# surf2=ax2.view_init(elev=20, azim=240)

# fig3,ax3=plt.subplots(subplot_kw={"projection":"3d"})
# surf3=ax3.plot_surface(testS1,testt1,err,linewidth=0,antialiased=False, cmap = 'viridis')
# surf3=ax3.set_xlabel('Harga Aset')
# surf3=ax3.set_ylabel('waktu')
# surf3=ax3.set_zlabel('Harga Opsi')
# surf3=ax3.view_init(elev=20, azim=240)
 