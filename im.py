# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:51 2017

@author: Jules Scholler
"""

import numpy as np
from scipy.signal import welch
from skimage.external.tifffile import TiffWriter
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.color import hsv2rgb, label2rgb, rgb2hsv
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import dilation, opening, remove_small_holes, remove_small_objects
from skimage.util import invert
from skimage.transform import rotate

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

def process_5_phases_OCT(data):
    dataAmp=np.zeros((int(data.shape[0]/5),data.shape[1],data.shape[2]))
    dataPhase=np.zeros((int(data.shape[0]/5),data.shape[1],data.shape[2]))
    for i in range(int(data.shape[0]/10)):
        u = data[i*10:(i+1)*10]
        I1 = u[0]
        I2 = u[1]
        I3 = u[2]
        I4 = u[3]
        I5 = u[4]
        I6 = u[5]
        I7 = u[6]
        I8 = u[7]
        I9 = u[8]
        I10 = u[9]
        dataAmp[2*i,:,:] = np.sqrt(4*(I2-I4)**2 + (I1-2*I3-I5)**2)
        dataAmp[2*i+1,:,:] = np.sqrt(4*(I9-I7)**2 + (I10-2*I8-I6)**2)
        dataPhase[2*i,:,:] = np.angle(np.sqrt(4*(I2-I4)**2-(I1-I5)**2) + 1j*(-I1+2*I3-I5))
        dataPhase[2*i+1,:,:] = np.angle(np.sqrt(4*(I9-I7)**2-(I10-I6)**2) + 1j*(-I10+2*I8-I6))
    return dataAmp,dataPhase


def fft(data, fs=2, n_piece=20):
    f = np.linspace(-fs/2, fs/2, data.shape[0])
    ind = (np.linspace(0, data.shape[1], n_piece)).astype('int')
    ind[-1] = data.shape[1]
    data_freq = np.zeros((f.size, data.shape[1], data.shape[2])) + 1j*np.zeros((f.size, data.shape[1], data.shape[2]))
    for i in range(n_piece-1):
        data_freq[:,ind[i]:ind[i+1],:] = np.abs(np.fft.fft(data[:,ind[i]:ind[i+1],:], axis=0))
    return (f,data_freq)

def fft_welch(data, fs=2, n_piece=20):
    f, a = welch(data[:,0,0], fs=2, window='flattop', scaling='density', nperseg=511)
    ind = (np.linspace(0, data.shape[1], n_piece)).astype('int')
    ind[-1] = data.shape[1]
    data_freq = np.zeros((f.size, data.shape[1], data.shape[2]))
    for i in range(n_piece-1):
        f, data_freq[:,ind[i]:ind[i+1],:] = welch(data[:,ind[i]:ind[i+1],:], fs=fs, window='flattop', scaling='density', axis=0, nperseg=511)
    return (f,data_freq)

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

def save_as_tiff(data, filename, resolution=(0.22,0.22)):
    savefile=(data/np.max(data)*(2**16-1)).astype('uint16')
    with TiffWriter(filename+'.tif', imagej=True) as tif:
        for i in range(savefile.shape[0]):
            tif.save(np.squeeze(savefile[i]), compress=0, resolution=resolution)
            
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

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def label_from_contour(img, contour):
    to_be_labeled = np.zeros(img.shape)
    for i, el in enumerate(contour):
        for ii in range(el.shape[0]):
            to_be_labeled[int(el[ii,0]),int(el[ii,1])] = 1
    return label(invert(dilation(to_be_labeled)), connectivity=1)

def label_from_list(img, regions):
    to_be_labeled = np.zeros(img.shape)
    for i, el in enumerate(regions):
        a = el['coords']
        for ii in range(a.shape[0]):
            to_be_labeled[int(a[ii,0]),int(a[ii,1])] = 1
    return label(to_be_labeled)

def butter2d_lp(shape, f, n, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (radius / f)**(2*n))
    return filt
 
def butter2d_bp(shape, cutin, cutoff, n, pxd=1):
    """Designs an n-th order bandpass 2D Butterworth filter with cutin and
   cutoff frequencies. pxd defines the number of pixels per unit of frequency
   (e.g., degrees of visual angle)."""
    return butter2d_lp(shape,cutoff,n,pxd) - butter2d_lp(shape,cutin,n,pxd)
 
def butter2d_hp(shape, f, n, pxd=1):
    """Designs an n-th order highpass 2D Butterworth filter with cutin
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    return 1. - butter2d_lp(shape, f, n, pxd)
 
def ideal2d_lp(shape, f, pxd=1):
    """Designs an ideal filter with cutoff frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = np.ones(shape)
    filt[radius>f] = 0
    return filt
 
def ideal2d_bp(shape, cutin, cutoff, pxd=1):
    """Designs an ideal filter with cutin and cutoff frequencies. pxd defines
   the number of pixels per unit of frequency (e.g., degrees of visual
   angle)."""
    return ideal2d_lp(shape,cutoff,pxd) - ideal2d_lp(shape,cutin,pxd)
 
def ideal2d_hp(shape, f, n, pxd=1):
    """Designs an ideal filter with cutin frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    return 1. - ideal2d_lp(shape, f, n, pxd)
 
def bandpass(data, highpass, lowpass, n, pxd, eq='histogram'):
    """Designs then applies a 2D bandpass filter to the data array. If n is
   None, and ideal filter (with perfectly sharp transitions) is used
   instead."""
    fft = np.fft.fftshift(np.fft.fft2(data))
    if n:
        H = butter2d_bp(data.shape, highpass, lowpass, n, pxd)
    else:
        H = ideal2d_bp(data.shape, highpass, lowpass, pxd)
    fft_new = fft * H
    new_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))    
    if eq == 'histogram':
        new_image = equalize_hist(new_image)
    return new_image

def remove_black_stripes(frames):
    for n,img in enumerate(frames):
        img = np.min(img, axis=0)
        bottom_left = []
        top_left = []
        bottom_right = []
        top_right = []
        
        stop = 0
        idx = 0
        while stop == 0:
            if np.mean(img[:,idx]) == 0:
                idx += 1
            else:
                stop = 1
                if np.min(np.where(img[:,idx]!=0)) > img.shape[0]/2:
                    bottom_left.append(np.max(np.where(img[:,idx]!=0)))
                    bottom_left.append(idx)
                else:
                    top_left.append(np.min(np.where(img[:,idx]!=0)))
                    top_left.append(idx)
                    
        stop = 0
        idx = 0
        while stop == 0:
            if np.mean(img[idx,:]) == 0:
                idx += 1
            else:
                stop = 1
                if np.min(np.where(img[idx,:]!=0)) < img.shape[1]/2:
                    top_left.append(idx)
                    top_left.append(np.min(np.where(img[idx,:]!=0)))
                else:
                    top_right.append(idx)
                    top_right.append(np.max(np.where(img[idx,:]!=0)))
                    
        stop = 0
        idx = img.shape[0]-1
        while stop == 0:
            if np.mean(img[idx,:]) == 0:
                idx -= 1
            else:
                stop = 1
                if np.min(np.where(img[idx,:]==0)) < img.shape[0]/2:
                    bottom_left.append(idx)
                    bottom_left.append(np.min(np.where(img[idx,:]!=0)))
                else:
                    bottom_right.append(idx)
                    bottom_right.append(np.max(np.where(img[idx,:]!=0)))
                    
        stop = 0
        idx = img.shape[1]-1
        while stop == 0:
            if np.mean(img[:,idx]) == 0:
                idx -= 1
            else:
                stop = 1
                if np.min(np.where(img[:,idx]!=0)) < img.shape[0]/2:
                    top_right.append(np.min(np.where(img[:,idx]!=0)))
                    top_right.append(idx)
                else:
                    bottom_right.append(np.max(np.where(img[:,idx]!=0)))
                    bottom_right.append(idx)
                    
        x1 = np.max([bottom_left[1], top_left[1]])
        y1 = np.max([top_left[0], top_right[0]])
        x2 = np.min([bottom_right[1], top_right[1]])
        y2 = np.min([bottom_left[0], bottom_right[0]])
        
        frames[n] = frames[n][:,y1:y2,x1:x2]
        
def plot_color(u,color,S = 0.8):
    im_hsv = np.zeros((*u.shape,3))
    im_hsv[:,:,2]=u-np.min(u);
    im_hsv[:,:,2]=im_hsv[:,:,2]/np.max(im_hsv[:,:,2]);
    im_hsv[:,:,1]=np.ones(u.shape) * S;
    im_hsv[:,:,0]=color-np.min(color);
    im_hsv[:,:,0]=im_hsv[:,:,0]/np.max(im_hsv[:,:,0])*0.66;
    im_rgb=hsv2rgb(im_hsv);
    return im_rgb

def copy_4image(u):
    s = np.array(u.shape)
    v = np.zeros((s[0]*2,s[1]*2))
    v[:s[0],:s[1]] = u
    v[:s[0],s[1]:] = np.fliplr(u)
    v[s[0]:,:s[1]] = np.flipud(u)
    v[s[0]:,s[1]:] = np.fliplr(np.flipud(u))
    return v
    
def measure_astigmatism(u, n):
    u_fft = np.abs(np.fft.fftshift(np.fft.fft2(copy_4image(u))))
    u_fft = gaussian(u_fft, sigma=3)
    u_fft = np.log(u_fft)
    s = u_fft.shape
    f = []
    for i in range(n):
        v_fft = np.zeros(u_fft.shape)
        u_fft_rotated = rotate(u_fft, angle=(i/n*360))
        v_fft[0:int(s[0]/2),0:int(s[1]/2)] = u_fft_rotated[0:int(s[0]/2),0:int(s[1]/2)]
        f.append(radial_profil(v_fft))
    ff = np.zeros((len(f),len(f[0])))
    for i,curve in enumerate(f):
        ff[i] = curve
    return np.std(ff, axis=0)

def compute_dffoct(u):
    """
    Compute D-FF-OCT image in grayspace with:
        - V: metabolic index (std with sliding window)
    
    """
    s = u.shape
    v = np.zeros((16, s[1], s[2]))
    for i in range(16):
       v[i] = np.std(u[i*32:(i+1)*32], axis=0)
    v = np.mean(v, axis=0)
    v_min, v_max = np.percentile(v.flatten(),(0.01, 99.9))
    return rescale_intensity(v, in_range=(v_min, v_max), out_range='float64')

def compute_dffoct_cumsum(u, n_sample, n_step):
    """
    Compute D-FF-OCT image in grayspace with:
        - V: metabolic index (cumsum with sliding window)
    
    """
    s = u.shape
    v = np.zeros((int(np.floor(s[0]/n_step)), s[1], s[2]))
    for i in range(int(np.floor(s[0]/n_step))):
       v[i] = np.std(np.add.accumulate(u[i*n_step:i*n_step+n_sample]-np.mean(u[i*n_step:i*n_step+n_sample], axis=0), axis=0),axis=0)
    v = np.mean(v, axis=0)
    v_min, v_max = np.percentile(v.flatten(),(0.01, 99.9))
    return rescale_intensity(v, in_range=(v_min, v_max), out_range='float64')

def compute_dffoct_cumsum_HSV(u, n_sample, n_step):
    """
    Compute D-FF-OCT image in the HSV space with:
        - V: metabolic index (cumsum with sliding window)
        - S: median frequency
        - H: frequency bandwidth
    """
    n_piece = 10
    fs = 1

    s = u.shape
    
    # Compute V
    V = compute_dffoct_cumsum(u, n_sample, n_step)
    
    # Compute Color (H component) with frequency median
    f, u_freq = fft_welch(u, fs=fs, n_piece=n_piece)
    s = u_freq.shape
    u_freq = preprocessing.normalize(u_freq.reshape((f.size,s[1]*s[2])), axis=0, norm='l1').reshape((f.size,s[1],s[2]))
    H = np.tensordot(u_freq[1:-1],np.linspace(0,fs,u_freq[1:-1].shape[0]),axes=(0,0))

    # Compute Saturation (S component) with frequency bandwidth
#    S = np.zeros((u.shape[1],u.shape[2]))
#    for i in range(cs.shape[1]):
#        for j in range(cs.shape[2]):
#            S[i,j] = np.sqrt(np.sum(u_freq[:,i,j]*f**2)-np.mean(u_freq[:,i,j])**2)
    S = np.ones((u.shape[1],u.shape[2]))*0.7
    
    v_min, v_max = np.percentile(H, (0.01, 99.9))
    Hf = rescale_intensity(H, in_range=(v_min, v_max), out_range='float')
    Hf = gaussian(Hf, sigma=3)
    Hf = rescale_intensity(1-Hf, out_range='float')*0.66
    
#    Sf = rescale_intensity(gaussian(1-S, sigma=2), out_range='float')
    dffoct_hsv = hsv2rgb(np.dstack((Hf,S,V)))
    return rescale_intensity(dffoct_hsv, out_range='uint8').astype('uint8')

def compute_dffoct_HSV(u):
    """
    Compute D-FF-OCT image in the HSV space with:
        - V: metabolic index (std with sliding window)
        - S: median frequency
        - H: frequency bandwidth
    
    """
    n_piece = 10
    fs = 1

    s = u.shape
    
    s = u.shape
    v = np.zeros((16, s[1], s[2]))
    for i in range(16):
       v[i] = np.std(u[i*32:(i+1)*32], axis=0)
    v = np.mean(v, axis=0)
    v_min, v_max = np.percentile(v.flatten(),(0.01, 99.9))
    V = rescale_intensity(v, in_range=(v_min, v_max), out_range='float64')
    
    f, u_freq = fft_welch(u, fs=fs, n_piece=n_piece)
    
    # Compute Color (H component) with frequency mean
    s = u_freq.shape
    u_freq = preprocessing.normalize(u_freq.reshape((f.size,s[1]*s[2])), axis=0, norm='l1').reshape((f.size,s[1],s[2]))
    H = np.tensordot(u_freq[1:-1],np.linspace(0,fs,u_freq[1:-1].shape[0]),axes=(0,0))

    # Compute Saturation (S component) with frequency bandwidth
#    S = np.zeros((u.shape[1],u.shape[2]))
#    for i in range(cs.shape[1]):
#        for j in range(cs.shape[2]):
#            S[i,j] = np.sqrt(np.sum(u_freq[:,i,j]*f**2)-np.mean(u_freq[:,i,j])**2)
    S = np.ones((u.shape[1],u.shape[2]))*0.66
    
    v_min, v_max = np.percentile(V, (1, 99.9))
    Vf = rescale_intensity(V, in_range=(v_min, v_max), out_range='float')
    v_min, v_max = np.percentile(H, (1, 99))
    Hf = rescale_intensity(H, in_range=(v_min, v_max), out_range='float')
    Hf = gaussian(Hf, sigma=2)
    Hf = rescale_intensity(-Hf, out_range='float')*0.6
    
#    Sf = rescale_intensity(gaussian(1-S, sigma=2), out_range='float')
    dffoct_hsv = hsv2rgb(np.dstack((Hf,S,Vf)))
    return rescale_intensity(dffoct_hsv, out_range='uint8').astype('uint8')

def ZCR(eigen_vect_t):
    """
    Compute zero crossing rate for temporal eigen vectors
    
    """
    v = []
    for i in range(eigen_vect_t.shape[0]):
        v.append(((eigen_vect_t[i,:-1]* eigen_vect_t[i,1:]) < 0).sum())
    return v

def preprocess_pmap(pmap):
    """
    Process probability map of cells to get boolean map.
    
    """
    thresh = 0.8
    pmap[pmap<thresh] = 0
    pmap[pmap>0] = 1
    t = opening(pmap, selem = np.ones((3,3)))
    t = t.astype(bool)
    t = remove_small_holes(t, 100)
    t = remove_small_objects(t, 100)
    return t
                
def get_cells_from_bool(t, dffoct, plot=False):
    """
    Get list of cells with properties from boolean map.
    
    """
    label_image = label(t)
    cells = []
    for region in regionprops(label_image):
        cells.append(region)
            
    if plot:
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
        ax[0].imshow(dffoct)
        ax[0].set_title('DFFOCT')
        image_label_overlay = label2rgb(label_image, image=dffoct, alpha=0.2, bg_label=0)
        ax[1].imshow(image_label_overlay)
        ax[1].set_title('Detected cells ' + str(len(cells)))
    return cells

def get_raw_signals_from_cells(cells, dffoct_raw):
    """
    Get dynamic signals from direct acquisition for each cells given in the list.
    
    """
    cells_dynamic = []
    for cnt,cell in enumerate(cells):
        coord = cell['coords']
        dyn = np.zeros((coord.shape[0],dffoct_raw.shape[0]))
        for ii in range(coord.shape[0]):
            dyn[ii] = dffoct_raw[:,int(coord[ii,0]),int(coord[ii,1])]-np.mean(dffoct_raw[:,int(coord[ii,0]),int(coord[ii,1])])
        cells_dynamic.append(dyn)
    return cells_dynamic

def stabilize_color(dffoct):
    """
    Stabilize color for timelapses and zStack of the shape [n_image, n_x, n_y, 3]
    It uses the first image as template for the rest.
    """
    tmp = rgb2hsv(dffoct[0])
    template = tmp[:,:,0]
    for i in range(dffoct.shape[0]-1):
        tmp = rgb2hsv(dffoct[i+1])
        tmp[:,:,0] = hist_match(tmp[:,:,0], template)
        tmp = hsv2rgb(tmp)
        tmp = rescale_intensity(tmp, out_range='uint8').astype('uint8')
        dffoct[i+1] = tmp
    return dffoct

def stabilize_running_color(dffoct,k):
    """
    Stabilize color for timelapses and zStack of the shape [n_image, n_x, n_y, 3].
    It uses a sliding window of size k to take into account normal changes in color/frequency.
    """
    n,nx,ny,_ = dffoct.shape
    for i in range(dffoct.shape[0]-k):
        h_template = rgb2hsv(dffoct[i:i+k].reshape((k*nx, ny, 3)))[:,:,0]
        tmp = rgb2hsv(dffoct[i])
        tmp[:,:,0] = hist_match(tmp[:,:,0], h_template)
        tmp = hsv2rgb(tmp)
        tmp = rescale_intensity(tmp, out_range='uint8').astype('uint8')
        dffoct[i] = tmp
    return dffoct