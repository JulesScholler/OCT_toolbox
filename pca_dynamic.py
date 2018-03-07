# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:02:43 2018

@author: Jules Scholler
"""

import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage.exposure import rescale_intensity
import OCT_toolbox.im as im
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.signal import welch

# Load data
direct = imread('direct.tif')
direct = direct[:,200:800,200:800]

# Normalize to prevent sensor drift effects
data=im.normalize(direct)

# Compute DFFOCT image
dffoct = im.DFFOCT_RGB(data, method='fft', fs=2, f1=0.05, f2=0.2, f3=0.5)

# Compute welch power spectrum
f,data_freq = welch(data, fs=300, nperseg=int(data.shape[0]/2), noverlap=int(data.shape[0]/4), axis=0)

# Reshape data to be time x voxels
s = data_freq.shape
data_freq = data_freq.reshape((s[0],s[1]*s[2]))
data_freq = data_freq.transpose() # X in PCA must be (n_samples, n_features)
data_freq = scale(data_freq, axis=0, with_mean=True, with_std=True)

# Process data
pca = PCA(n_components=s[0], svd_solver='full')
data_pca = pca.fit_transform(data_freq) # data_pca is (n_samples, n_components)
data_pca = data_pca.transpose() # data_pca is (n_components, n_samples)
data_pca = data_pca.reshape(s)

u = np.zeros(s)
for i in range(s[0]):
    u[i] = rescale_intensity(data_pca[i], out_range='uint8')

im.save_as_tiff(u, 'pca_welch_standardized')

explained = pca.explained_variance_ratio_
plt.figure()
plt.semilogy(explained)

components = pca.components_

n=5

fig, ax = plt.subplots(n,1)
for i in range(n):
    ax[i].plot(f,components[i,:])
    
# Reconstruct data
data_pca = data_pca.reshape((s[0],s[1]*s[2]))
data_pca = data_pca.transpose() # data_pca is (n_components, n_samples)
new_data_pca = np.zeros(data_pca.shape)
new_data_pca[:,1:3] = data_pca[:,1:3]
new_data = pca.inverse_transform(new_data_pca)

