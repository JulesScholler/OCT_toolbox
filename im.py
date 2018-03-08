# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:51 2017

@author: Jules Scholler
"""

import numpy as np
from scipy.signal import welch
from skimage.external.tifffile import TiffWriter
from skimage.exposure import rescale_intensity, equalize_adapthist, histogram
from skimage.color import hsv2rgb
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

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

def DFFOCT_HSV(data, method='fft', fs=2, n_mean=10):
    """
    Compute D-FF-OCT image in the HSV space with:
        - V: metabolic index (std with sliding window)
        - S: median frequency
        - H: frequency bandwidth
    
    """
    if method=='welch':
        # We divide the input in 4 pieces in order to avoid memory troubles
        f, a = welch(data[:,0,0], fs=2, window='flattop', scaling='density')
        dx = int(data.shape[1]/2)
        dy = int(data.shape[2]/2)
        data_freq = np.zeros((f.size, data.shape[1], data.shape[2]))
        f, data_freq[:, 0:dx, 0:dy] = welch(data[:, 0:dx, 0:dy], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, 0:dx, dy:data.shape[2]] = welch(data[:, 0:dx, dy:data.shape[2]], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, dx:data.shape[1], 0:dy] = welch(data[:, dx:data.shape[1], 0:dy], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, dx:data.shape[1], dy:data.shape[2]] = welch(data[:, dx:data.shape[1], dy:data.shape[2]], fs=2, window='flattop', scaling='density', axis=0)

    elif method=='fft':
        # We divide the input in 4 pieces in order to avoid memory troubles
        dx = int(data.shape[1]/2)
        dy = int(data.shape[2]/2)
        s = data.shape
        data_freq = np.zeros(s)
        data = preprocessing.scale(np.reshape(data,(s[0],s[1]*s[2])), axis=0, with_std=False).reshape(s)
        data_freq[:, 0:dx, 0:dy] = np.abs(np.fft.fft(data[:, 0:dx, 0:dy], axis=0))
        data_freq[:, 0:dx, dy:data.shape[2]] = np.abs(np.fft.fft(data[:, 0:dx, dy:data.shape[2]], axis=0))
        data_freq[:, dx:data.shape[1], 0:dy] = np.abs(np.fft.fft(data[:, dx:data.shape[1], 0:dy], axis=0))
        data_freq[:, dx:data.shape[1], dy:data.shape[2]] = np.abs(np.fft.fft(data[:, dx:data.shape[1], dy:data.shape[2]], axis=0))
        data_freq = data_freq[0:np.floor(s[0]/2).astype('int')]
        f = np.linspace(0, fs/2, data_freq.shape[0])
        
    else:
        print('Unknown method')
        
    # Compute Intensity (V component) with substacks STD
    n_substack = data.shape[0]-n_mean
    a = np.zeros((n_substack,data.shape[1],data.shape[2]))
    for i in range(n_substack):
        a[i] = np.std(data[i:i+n_mean], axis=0)
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

    V = rescale_intensity(V,out_range='float')
    H = rescale_intensity(H,out_range='float')
    S = rescale_intensity(S,out_range='float')
    return hsv2rgb(np.dstack((H,S,V)))

def DFFOCT_RGB(data, method='fft', fs=2, f1=0.1, f2=0.5, f3=1):
    """
    Compute D-FF-OCT image in the RGB space with:
        - R: high frequencies
        - G: medium frequencies
        - B: low frequencies
    
    """
    if method=='welch':
        # We divide the input in 4 pieces in order to avoid memory troubles
        f, a = welch(data[:,0,0], fs=2, window='flattop', scaling='density')
        dx = int(data.shape[1]/2)
        dy = int(data.shape[2]/2)
        data_freq = np.zeros((f.size, data.shape[1], data.shape[2]))
        f, data_freq[:, 0:dx, 0:dy] = welch(data[:, 0:dx, 0:dy], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, 0:dx, dy:data.shape[2]] = welch(data[:, 0:dx, dy:data.shape[2]], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, dx:data.shape[1], 0:dy] = welch(data[:, dx:data.shape[1], 0:dy], fs=2, window='flattop', scaling='density', axis=0)
        f, data_freq[:, dx:data.shape[1], dy:data.shape[2]] = welch(data[:, dx:data.shape[1], dy:data.shape[2]], fs=2, window='flattop', scaling='density', axis=0)

    elif method=='fft':
        # We divide the input in 4 pieces in order to avoid memory troubles
        dx = int(data.shape[1]/2)
        dy = int(data.shape[2]/2)
        s = data.shape
        data_freq = np.zeros(s)
        data = preprocessing.scale(np.reshape(data,(s[0],s[1]*s[2])), axis=0, with_std=False).reshape(s)
        data_freq[:, 0:dx, 0:dy] = np.abs(np.fft.fft(data[:, 0:dx, 0:dy], axis=0))
        data_freq[:, 0:dx, dy:data.shape[2]] = np.abs(np.fft.fft(data[:, 0:dx, dy:data.shape[2]], axis=0))
        data_freq[:, dx:data.shape[1], 0:dy] = np.abs(np.fft.fft(data[:, dx:data.shape[1], 0:dy], axis=0))
        data_freq[:, dx:data.shape[1], dy:data.shape[2]] = np.abs(np.fft.fft(data[:, dx:data.shape[1], dy:data.shape[2]], axis=0))
        data_freq = data_freq[0:np.floor(s[0]/2).astype('int')]
        f = np.linspace(0, fs/2, data_freq.shape[0])
        
    else:
        print('Unknown method')
    
    fs1 = int(np.round(2*f1/fs*data_freq.shape[0]))
    fs2 = int(np.round(2*f2/fs*data_freq.shape[0]))
    fs3 = int(np.round(2*f3/fs*data_freq.shape[0]))
    R = np.log(np.sum(data_freq[fs2:,:,:], axis=0))
    G = np.log(np.sum(data_freq[fs1:fs2,:,:], axis=0))
    B = np.log(np.sum(data_freq[0:fs1,:,:], axis=0))

    R = rescale_intensity(R,out_range='float')
    G = rescale_intensity(G,out_range='float')
    B = rescale_intensity(B,out_range='float')
    
    return np.dstack((R,G,B))

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