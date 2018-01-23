# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:51 2017

@author: Jules Scholler
"""

import numpy as np
from scipy.signal import welch
from skimage.external.tifffile import TiffWriter

def normalize(data):
    """ The camera sensor drift and we correct it by normalizing each image by the total energy. """
    imNorm=np.zeros(data.shape)
    for i in range(data.shape[0]):
        imNorm[i]=data[i]/np.mean(data[i])
    del(data)
    # Rescale between 0 and 2**16
    imNorm=imNorm-np.min(imNorm)
    imNorm=imNorm/np.max(imNorm)
    imNorm=imNorm*2**16
    imNorm=imNorm.astype('uint16')
    return imNorm

def average(data,n=10):
    # Compute images average in a stack with n images
    imAv=np.zeros((int(np.floor(data.shape[0]/n)),data.shape[1],data.shape[2]))
    for i in range(int(np.floor(data.shape[0]/n))):
        imAv[i,:,:]=np.mean(data[i*n:(i+1)*n,:,:],axis=0)
    imAv=imAv.astype('uint16')
    return imAv

def std(data,n=10):
    # Compute images STD in a stack with n images
    imSTD=np.zeros((data.shape[0]-n,data.shape[1],data.shape[2]))
    for i in range(data.shape[0]-n):
        imSTD[i,:,:]=np.std(data[i:i+n,:,:],axis=0)
    return imSTD

def process_2_phases_OCT(data):
    dataAmp=np.zeros((int(data.shape[0]/2),data.shape[1],data.shape[2]))
    for i in range(int(data.shape[0]/2)):
        dataAmp[i,:,:]=np.abs(data[2*i]-data[2*i+1])
    return dataAmp

def process_4_phases_OCT(data):
    dataAmp=np.zeros((int(data.shape[0]/4),data.shape[1],data.shape[2]))
    dataPhase=np.zeros((int(data.shape[0]/4),data.shape[1],data.shape[2]))
    for i in range(int(data.shape[0]/4)):
        dataAmp[i,:,:]=0.5*np.sqrt((data[4*i+4]-data[4*i+2])**2+(data[4*i+3]-data[4*i+1])**2)
        dataPhase[i,:,:]=np.angle((data[4*i+3]-data[4*i+1])/(data[4*i+4]-data[4*i+2]))
    return dataAmp,dataPhase

def DFFOCT(data,method='fft'):
    imD=np.zeros((data.shape[1],data.shape[2],3))
    if method=='welch':
        # Compute welch periodogram
        t=1
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                f, a = welch(data[:,i,j], fs=2, window='flattop', scaling='density')
                imD[i,j,0]=np.sum(a[1:int(a.shape[0]/3)]**2)
                imD[i,j,1]=np.sum(a[int(a.shape[0]/3)+1:2*int(a.shape[0]/3)]**2)
                imD[i,j,2]=np.sum(a[2*int(a.shape[0]/3)+1:3*int(a.shape[0]/3)]**2)
                if i*j/(data.shape[1]*data.shape[2])>t/100:
                    print(t,'%')
                    t=t+1
        imD=np.log10(imD)
        imD=imD-np.min(imD)
        imD=imD/np.max(imD)*255
        imD=imD.astype('uint8')
    elif method=='fft':
        # Compute fft
        t=1
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                a=np.abs(np.fft.fft(data[:,i,j]))
                imD[i,j,0]=np.sum(a[1:int(a.shape[0]/3)]**2)
                imD[i,j,1]=np.sum(a[int(a.shape[0]/3)+1:2*int(a.shape[0]/3)]**2)
                imD[i,j,2]=np.sum(a[2*int(a.shape[0]/3)+1:3*int(a.shape[0]/3)]**2)
                if i*j/(data.shape[1]*data.shape[2])>t/100:
                    print(t,'%')
                    t=t+1
        imD=np.log10(imD)
        imD=imD-np.min(imD)
        imD=imD/np.max(imD)*255
        imD=imD.astype('uint8')
    elif method=='metabolic':
        print('not supported yet')
    else:
        print('Unknown method')
    return imD

def radial_profil(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def pfft2(im): 
    [rows,cols] = im.shape
    #Compute boundary conditions 
    s = np.zeros( im.shape ) 
    s[0,0:] = im[0,0:] - im[rows-1,0:] 
    s[rows-1,0:] = -s[0,0:] 
    s[0:,0] = s[0:,0] + im[0:,0] - im[:,cols-1] 
    s[0:,cols-1] = s[0:,cols-1] - im[0:,0] + im[:,cols-1] 
    #Create grid for computing Poisson solution 
    [cx, cy] = np.meshgrid(2*np.pi*np.arange(0,cols)/cols, 2*np.pi*np.arange(0,rows)/rows) 
    
    #Generate smooth component from Poisson Eq with boundary condition 
    D = (2*(2 - np.cos(cx) - np.cos(cy))) 
    D[0,0] = np.inf # Enforce 0 mean & handle div by zero 
    S = np.fft.fft2(s)/D 
    
    P = np.fft.fft2(im) - S # FFT of periodic component 
    return P

def online_std(data,M1,mean1,n):
#    mean2=mean1+(data-mean1)/n
#    M2=M1+(data-mean1)*(data-mean2)
#    var2=M2/(n-1)
    if n==0:
        mean1 = 0.0
        M1 = 0.0

    delta = data - mean1
    mean2 = mean1 + delta/(n+1)
    delta2 = data - mean2
    M2 = M1 + delta*delta2
    return mean2,M2

def save_as_tiff(data):
    savefile=(data/np.max(data)*(2**16-1)).astype('uint16')
    with TiffWriter('zStack_denoised.tif', imagej=True) as tif:
        for i in range(savefile.shape[0]):
            tif.save(savefile[i], compress=0)