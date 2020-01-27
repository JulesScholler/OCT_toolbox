# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:30:10 2017

@author: Jules Scholler
"""

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import label2rgb
import numpy as np
import cv2
from matplotlib.gridspec import GridSpec
from scipy.signal import welch

def img_and_hist(image, bins=256):

    ax_img = plt.subplot(1, 2, 1)
    ax_hist = plt.subplot(1, 2, 2)
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin=image.min()
    xmax=image.max()
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    
def plot_time_series(imSTDtot,imNorm,n):
    
    plt.figure()
    gs = GridSpec(3,2)
    
    axDFFOCT = plt.subplot(gs[:,0])
    axDFFOCT.set_title('Dynamic image')
    axDFFOCT.set_axis_off()
    
    axTime = plt.subplot(gs[0,1])
    axTime.set_title('Time series')
    axTime.set_xlabel('Time sample [#]')
    axTime.set_ylabel('Intensity [a.u.]')
    
    axPSD = plt.subplot(gs[1,1])
    axPSD.set_title('Power spectrum density')
    axPSD.set_xlabel('Frequency [Hz]')
    axPSD.set_ylabel('PSD [$V^2/Hz$]')
    
    axXcorr = plt.subplot(gs[2,1])
    axXcorr.set_title('Auto-correlation')
    axXcorr.set_xlabel('Delay [#]')
    axXcorr.set_ylabel('Normalized auto-correlation')
    
    i = 0
    while 1:
        axDFFOCT.imshow(imSTDtot)

        x=plt.ginput(1)
        x=np.array(x)
        x=x.astype(int)
        dta = imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]])
        
        axDFFOCT.plot(x[0,0], x[0,1], '+r')
        axTime.plot(dta)
#        f,p = welch(imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]), nperseg=512)
        f = np.cumsum(dta)
        axPSD.plot(f)
        u = np.correlate(imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]),imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]), 'full')/np.dot(imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]),imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]))
        axXcorr.plot(u[int(u.size/2):])
        plt.pause(0.05)
        if i>=n-1:
            axTime.lines[0].remove()
            axPSD.lines[0].remove()
            axDFFOCT.lines[0].remove()
            axXcorr.lines[0].remove()
        i += 1

def overlay(u,v,alpha):
    im1=np.zeros((u.shape[0],u.shape[1],3))
    im2=im1.copy()
    for i in range(3):
        im1[:,:,i]=u[0]/np.max(u)
    im1=np.log(im1+0.0001)
    im1=im1-np.min(im1)
    im1=(im1/np.max(im1)*255).astype('uint8')
    im2[:,:,0]=(v*255).astype('uint8')
    a = (im2[:,:,0] == 0)*(im2[:,:,1] == 0)*(im2[:,:,2] == 0)
    im2[a]=255
    im2=im2.astype('uint8')
    image_overlay=cv2.addWeighted(im1, 1-alpha, im2, alpha, 0)
    plt.imshow(image_overlay)
    
def plot_features_2d(features, scale):
    '''
    This function plots n features against each other, the number of plots is (n-1)*n/2
    '''
    n = features.shape[1]
    fig, ax = plt.subplots(n-1,n-1)
    for i in range(n):
        for j in range(n):
            if j>i:
                ax[i,j-1].plot(features[0::scale,i],features[0::scale,j], 'k+')
                
def plot_cells_from_list(cells, dffoct):
    label_image = np.zeros(dffoct.shape)
    for cnt, cell in enumerate(cells):
        coord = cell['coords']
        for ii in range(coord.shape[0]):
            label_image[coord[ii,0],coord[ii,1]] = cnt
    plt.figure()
    plt.imshow(label2rgb(label_image, image=dffoct, alpha=0.2, bg_label=np.min(label_image[:])))