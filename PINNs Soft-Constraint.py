#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:25:58 2023

@author: dennisirwanto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:00:13 2023

@author: dennisirwanto
"""

'''
ALGORITMA PHYSICS-INFORMED NEURAL NETWORKS UNTUK PDP BLACK-SCHOLES NORMALISASI
'''

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from scipy.interpolate import griddata
import scipy.stats as stats
import tensorflow as tf


K = 100  # Strike price
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility

# Define the Black-Scholes PDE
def black_scholes_pde(x, y):
    S, t = x[:, 0:1], x[:, 1:2]
    K = 0.5  # Strike price
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility
     
    f = y
    S, t = x[:, 0:1], x[:,1:2]
    T = tf.reduce_max(t)
     
    df_t = dde.grad.jacobian(y, x, i=0, j=1)
    df_S = dde.grad.jacobian(y, x, i=0, j=0)
    df_SS = dde.grad.hessian(y, x, i=0, j=0)
    return df_t + (1/2) * (sigma**2) * S**2 * df_SS + r * S * df_S - r * f 

# Define the boundary condition
def func(x):
    K = 0.5
    S = x[:, 0:1]
    return tf.maximum(S-K,0)

def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


# Define the geometry and time domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# Define the boundary condition
bc1 = dde.icbc.DirichletBC(geom, func, boundary_left)
bc2 = dde.icbc.DirichletBC(geom, func, boundary_right)
ic = dde.icbc.DirichletBC(geom, func, boundary_top)
#bc1 = dde.icbc.boundary_conditions.DirichletBC(geomtime, func, boundary_condition_1)
#bc2 = dde.icbc.boundary_conditions.DirichletBC(geomtime, func, boundary_condition_2)
#ic = dde.icbc.boundary_conditions.DirichletBC(geomtime, func, final_condition)

# Define the data object
data = dde.data.PDE(geom, 
                        black_scholes_pde, 
                        [bc1, bc2, ic], 
                        num_domain=850,
                        num_boundary=150,
                        num_test=150)

# Define the neural network model
layer_size = [2] + [16] * 8 + [1]
activation = "sigmoid"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
#net.apply_output_transform(lambda x, y: abs(y))

# Define the model
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=0.001)
model.train(epochs=10000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# plt.figure()
# plt.semilogy(losshistory.loss_train, label="Train Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss (Log Scale)")
# plt.legend(
# plt.title("Training Loss History")
# plt.grid(True)
# plt.show()

testS = np.linspace(0, 1, 100)
testt = np.linspace(0, 1, 100)
test_S, test_t = np.meshgrid(testS, testt)

testS2= np.linspace(0, 200, 100)
test_S2, test_t2= np.meshgrid(testS2, testt)

max_test_T = np.max(test_t2) # waktu jatuh tempo T = 1
max_test_S2 = np.max(test_S2)

S_star = np.hstack((test_S.flatten()[:, None], test_t.flatten()[:, None]))
prediction_f = model.predict(S_star, operator=None)

pred_f = griddata(S_star, prediction_f[:, 0], (test_S, test_t), method="cubic")
pred_f = pred_f * max_test_S2 

fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
surf=ax.plot_surface(test_S2,test_t2,pred_f,linewidth=0,antialiased=False, cmap = 'viridis')
surf=ax.set_xlabel('Harga Aset')
surf=ax.set_ylabel('waktu')
surf=ax.set_zlabel('Harga Opsi')
surf=ax.view_init(elev=20, azim=240)

# Visualize the option prices
# You can use matplotlib or any other library to visualize the results.
# For example, create a 3D plot of option prices over time and asset prices.

# Mendefinisikan solusi analitik
def solusi_analitik_call(S, K, T, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    f = S * N_d1 - K * np.exp(-r * (T-t)) * N_d2
    return f

test_Sa=test_S2.reshape(-1)
test_ta=test_t2.reshape(-1)

max_test_T=np.max(test_ta)

true_f = solusi_analitik_call(test_Sa, K, max_test_T, test_ta, r, sigma)
true_f = true_f 
true_f = true_f.reshape((100,100))

fig2,ax2=plt.subplots(subplot_kw={"projection":"3d"})
surf2=ax2.plot_surface(test_S2,test_t2,true_f,linewidth=0,antialiased=False, cmap = 'viridis')
surf2=ax2.set_xlabel('Harga Aset')
surf2=ax2.set_ylabel('waktu')
surf2=ax2.set_zlabel('Harga Opsi')
surf2=ax2.view_init(elev=20, azim=240)

#Penyebaran error antara nilai hasil prediksi neural network dengan solusi analitik
err=pred_f-true_f

fig3,ax3=plt.subplots(subplot_kw={"projection":"3d"})
surf3=ax3.plot_surface(test_S2,test_t2,err,linewidth=0,antialiased=False, cmap = 'viridis')
surf3=ax3.set_xlabel('Harga Aset')
surf3=ax3.set_ylabel('waktu')
surf3=ax3.set_zlabel('Harga Opsi')
surf3=ax3.view_init(elev=20, azim=240)

fig4,ax4=plt.subplots()
surf4=ax4.pcolormesh(test_S2,test_t2,err,cmap = 'viridis')
fig4.colorbar(surf4, label='Penyebaran Error')
surf4=ax4.set_xlabel('Harga Aset')
surf4=ax4.set_ylabel('waktu')
#surf3=ax3.view_init(elev=20, azim=240)

########## SLICE TEST PADA t=0
test_t3= 0

#test_S= train_S * max_train_S_accent                                 # Membuat 100 test x, dengan range yg sama yaitu x- [0,1]
#test_t= max_test_T - train_t * max_test_T

max_test_T = np.max(test_t) # waktu jatuh tempo T = 1
                                

testS4,testt3 = np.meshgrid(testS2, test_t3)
testSd=testS4.reshape(-1)
testtd=testt3.reshape(-1)

predf_t0 = pred_f[0,:].reshape((100,1))

true_f_t0 = true_f[0,:].reshape((100,1))

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

