# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:51 2017

@author: Jules Scholler
"""

import numpy as np
from scipy.signal import welch
from skimage.external.tifffile import TiffWriter
from skimage.exposure import rescale_intensity
from skimage.color import hsv2rgb
from skimage.filters import gaussian
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import matplotlib.pyplot as plt
import cv2


def normalize(data):
    """ The camera sensor drift and we correct it by normalizing each image by the total energy. """
    directMean = np.mean(data, axis=(1,2))
    directMax = np.max(2**16/directMean)
    c = (2**16-1)/directMax
    imNorm=np.zeros(data.shape).astype('uint16')
    for i in range(data.shape[0]):
        imNorm[i]=(data[i]/directMean[i]*c).astype('uint16')
    return imNorm

def average(data,n=10):
    """ Compute images average in a stack with n images. """
    imAv=np.zeros((int(np.floor(data.shape[0]/n)),data.shape[1],data.shape[2]))
    for i in range(int(np.floor(data.shape[0]/n))):
        imAv[i,:,:]=np.mean(data[i*n:(i+1)*n,:,:],axis=0)
    imAv=imAv.astype('uint16')
    return imAv

def std(data,n=10):
    """ Compute images std in a stack with n images. """
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

def fft_welch(data, fs=2, n_piece=20):
    f, a = welch(data[:,0,0], fs=2, window='flattop', scaling='density')
    ind = (np.linspace(0, data.shape[1], n_piece)).astype('int')
    ind[-1] = data.shape[1]
    data_freq = np.zeros((f.size, data.shape[1], data.shape[2]))
    for i in range(n_piece-1):
        f, data_freq[:,ind[i]:ind[i+1],:] = welch(data[:,ind[i]:ind[i+1],:], fs=fs, window='flattop', scaling='density', axis=0)
    return (f,data_freq)

def fft(data, fs=2, n_piece=20):
    f = np.linspace(-fs/2, fs/2, data.shape[0])
    ind = (np.linspace(0, data.shape[1], n_piece)).astype('int')
    ind[-1] = data.shape[1]
    data_freq = np.zeros((f.size, data.shape[1], data.shape[2])) + 1j*np.zeros((f.size, data.shape[1], data.shape[2]))
    for i in range(n_piece-1):
        data_freq[:,ind[i]:ind[i+1],:] = np.abs(np.fft.fft(data[:,ind[i]:ind[i+1],:], axis=0))
    return (f,data_freq)

def DFFOCT_HSV(data, fs=2, n_std=50, n_piece=20):
    """
    Compute D-FF-OCT image in the HSV space with:
        - V: metabolic index (std with sliding window)
        - S: median frequency
        - H: frequency bandwidth
    
    """
    
    f, data_freq = fft_welch(data, fs=fs, n_piece=n_piece)
    
    # Compute Intensity (V component) with substacks STD
    n_substack = data.shape[0]-n_std
    a = np.zeros((n_substack,data.shape[1],data.shape[2]))
    for i in range(n_substack):
        a[i] = np.std(data[i:i+n_std], axis=0)
    V = np.mean(a, axis=0)
    del(a)
    
    # Compute Color (H component) with frequency median
    s = data_freq.shape
    data_freq = preprocessing.normalize(data_freq.reshape((f.size,s[1]*s[2])), axis=0, norm='l1').reshape((f.size,s[1],s[2]))
    cs = np.cumsum(data_freq, axis=0)
    H = np.zeros((data.shape[1],data.shape[2]))
    for i in range(cs.shape[1]):
        for j in range(cs.shape[2]):
            H[i,j] = np.min(np.where(cs[:,i,j]>=0.5))
    
    # Compute Saturation (S component) with frequency bandwidth
    S = np.zeros((data.shape[1],data.shape[2]))
    for i in range(cs.shape[1]):
        for j in range(cs.shape[2]):
            S[i,j] = np.sqrt(np.sum(data_freq[:,i,j]*f**2)-np.mean(data_freq[:,i,j])**2)
    
    v_min, v_max = np.percentile(V, (0, 99.9))
    Vf = rescale_intensity(V, in_range=(v_min, v_max), out_range='float')
    Hf = rescale_intensity(gaussian(H, sigma=3), out_range='float')
    Sf = rescale_intensity(gaussian(1-S, sigma=2), out_range='float')
    dffoct_hsv = hsv2rgb(np.dstack((Hf,Sf,Vf)))
    return rescale_intensity(dffoct_hsv, out_range='uint8').astype('uint8')

def DFFOCT_RGB(data, method='fft', fs=2, n_mean=4, n_piece=20):
    """
    Compute D-FF-OCT image in the RGB space with:
        - R: high frequencies
        - G: medium frequencies
        - B: low frequencies
    Time dimension must be on axis 0.
    
    """
    if n_mean>0:
        data = average(data, n=n_mean)
    if method=='welch':
        f, data_freq = fft_welch(data, fs=fs, n_piece=n_piece)
    elif method=='fft':
        f, data_freq = fft(data, fs=fs, n_piece=n_piece)
    else:
        print('Unknown method')
    
    B = data_freq[1,:,:]
    v_min, v_max = np.percentile(B, (1, 99))
    B = rescale_intensity(B, in_range=(v_min, v_max), out_range='uint8')
    
    G = np.mean(data_freq[2:17,:,:], axis=0)
    v_min, v_max = np.percentile(G, (1, 99))
    G = rescale_intensity(G, in_range=(v_min, v_max), out_range='uint8')
    
    R = np.mean(data_freq[18:np.min((80,data_freq.shape[0]-1)),:,:], axis=0)
    v_min, v_max = np.percentile(R, (1, 95))
    R = rescale_intensity(R, in_range=(v_min, v_max), out_range='uint8')
    
    return np.dstack((R,G,B)).astype('uint8')

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

def save_as_tiff(data, filename):
    savefile=(data/np.max(data)*(2**16-1)).astype('uint16')
    with TiffWriter(filename+'.tif', imagej=True) as tif:
        for i in range(savefile.shape[0]):
            tif.save(savefile[i], compress=0)
            
def write_text(imPath, text, position=(0,0), color=(255,255,255), size=100):
    a = Image.open(imPath)
    draw = ImageDraw.Draw(a)
    font = ImageFont.truetype("arial.ttf",100)
    draw.text(position, text, color, font=font)
    a.save(imPath[0:-4] + '_text.tif')
    
def build_colormap(Lx=50, Ly=1000, fs=2):
    color = np.linspace(1,0,Ly)
    H = np.matlib.repmat(color,Lx,1).transpose()
    
    saturation = np.linspace(1, 0, Lx)
    S = np.matlib.repmat(saturation, Ly, 1)
    V = np.ones((Ly,Lx))*0.7
    
    colormap = {}
    colormap['image'] = hsv2rgb(np.dstack((H,S,V)))
    colormap['freq'] = np.linspace(0,fs/2,Ly)
    colormap['df'] = np.linspace(0,fs/2,Lx)
    
    dx = (colormap['df'][1]-colormap['df'][0])/2.
    dy = (colormap['freq'][1]-colormap['freq'][0])/2.
    extent = [colormap['df'][0]-dx, colormap['df'][-1]+dx, colormap['freq'][0]-dy, colormap['freq'][-1]+dy]
    
    plt.imshow(colormap['image'], extent=extent)
    plt.xlabel('Frequency bandwidth [Hz]')
    plt.ylabel('Median frequency [Hz]')
    
    return colormap

def overlay(u, v, alpha=0.5, plot=0):
    im1 = np.zeros((max(u.shape[0],v.shape[0]),max(u.shape[1],v.shape[1]),3))
    im2 = im1.copy()
    for i in range(3):
        im1[0:u.shape[0],0:u.shape[1],i] = u/np.max(u)
    im1 = (im1/np.max(im1)*255).astype('uint8')
    im2[0:v.shape[0],0:v.shape[1],0] = (v/np.max(v)*255)
    im2 = im2.astype('uint8')
    image_overlay = cv2.addWeighted(im1, 1-alpha, im2, alpha, 0)
    if plot==1:
        plt.imshow(image_overlay)
    return image_overlay.astype('uint8')