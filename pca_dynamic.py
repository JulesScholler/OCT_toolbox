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
import OCT_toolbox.util as util
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.signal import welch

# Load data
#direct = util.loadmatv7(r'C:\Users\Jules Scholler\Desktop\LOUIS\manip in vivo\direct.mat')
#direct = direct['im']
direct = imread(r'C:\Users\Jules Scholler\Desktop\LOUIS\ex vivo\direct.tif')

# Normalize to prevent sensor drift effects
data=im.normalize(direct)
del(direct)

# Compute DFFOCT image
dffoct = im.DFFOCT_RGB(data, method='fft')

# Compute welch power spectrum
data = im.average(data, n=4)
f, a = welch(data[:,0,0], fs=2, window='flattop', scaling='density')
dx = int(data.shape[1]/2)
dy = int(data.shape[2]/2)
data_freq = np.zeros((f.size, data.shape[1], data.shape[2]))
f, data_freq[:, 0:dx, 0:dy] = welch(data[:, 0:dx, 0:dy], fs=300, window='flattop', scaling='density', axis=0)
f, data_freq[:, 0:dx, dy:data.shape[2]] = welch(data[:, 0:dx, dy:data.shape[2]], fs=300, window='flattop', scaling='density', axis=0)
f, data_freq[:, dx:data.shape[1], 0:dy] = welch(data[:, dx:data.shape[1], 0:dy], fs=300, window='flattop', scaling='density', axis=0)
f, data_freq[:, dx:data.shape[1], dy:data.shape[2]] = welch(data[:, dx:data.shape[1], dy:data.shape[2]], fs=300, window='flattop', scaling='density', axis=0)


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

im.save_as_tiff(u, 'pca_welch_no_processings')

explained = pca.explained_variance_ratio_
plt.figure()
plt.semilogy(explained)
plt.xlabel('PCA Component [#]')
plt.ylabel('Ratio of explained variance')

components = pca.components_
n=5
fig, ax = plt.subplots(n,1)
for i in range(n):
    ax[i].plot(f,components[i,:])
    ax[i].set_ylabel('PC n° %d' % i)
ax[n-1].set_xlabel('Feature sample [#]')

        
# Reconstruct data
data_pca = data_pca.reshape((s[0],s[1]*s[2]))
data_pca = data_pca.transpose() # data_pca is (n_components, n_samples)
new_data_pca = np.zeros(data_pca.shape)
new_data_pca[:,1:3] = data_pca[:,1:3]
new_data = pca.inverse_transform(new_data_pca)
# Compute Higher order decomposition using Tucker method


