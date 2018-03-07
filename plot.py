# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:30:10 2017

@author: Jules Scholler
"""

import matplotlib.pyplot as plt
from skimage.util.dtype import dtype_range
from skimage import exposure
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
    
def plot_time_series(imSTDtot,imNorm):
    
    plt.figure()
    gs = GridSpec(2,2)
    
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
    
    plt.ion()

    while 1:
        axDFFOCT.imshow(imSTDtot)
        print('Click on the position to plot the time serie\n')
        x=plt.ginput(1)
        x=np.array(x)
        x=x.astype(int)
        
        # Compute STD for different film length
        axTime.plot(imNorm[:,x[0,1],x[0,0]]-np.mean(imNorm[:,x[0,1],x[0,0]]))
        f,p = welch(imNorm[:,x[0,1],x[0,0]], nperseg=512)
        axPSD.plot(f,p)
        plt.pause(0.05)

def multi_slice_viewer(volume,sat):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index],vmax=sat)
    fig.suptitle(ax.index+1, fontsize=16)
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax,fig)
    elif event.key == 'k':
        next_slice(ax,fig)
    fig.canvas.draw()

def previous_slice(ax,fig):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    fig.suptitle(ax.index+1, fontsize=16)

def next_slice(ax,fig):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    fig.suptitle(ax.index+1, fontsize=16)

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
    
def img_nav(img_stack):
    '''
    Function to conveniently navigate through a stack of images (like img_outputs) with the keyboard arrows
    Can be placed inside a relevant function folder...
    '''
#    curr_pos = 0
    class ChangeFig:
        def __init__(self,n,img,img_stack):
            self.curr_pos = 0
            self.img_stack = img_stack
            self.length= n
            self.img = img
            print('here')
            self.cid = img.figure.canvas.mpl_connect('key_press_event',self)
            
        def __call__(self,event):
            
            if event.key == "right":
                self.curr_pos = self.curr_pos + 1
            elif event.key == "left":
                self.curr_pos = self.curr_pos - 1
            self.curr_pos = self.curr_pos % self.length

#            print('image no : %d'%self.curr_pos)

            self.img.set_data(self.img_stack[self.curr_pos])
            self.img.figure.canvas.draw()
            self.img.axes.set_title("Image number : %d" % self.curr_pos)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(img_stack[0],vmin = 0, vmax = np.max(img_stack),cmap = 'hot')
    ax.set_title("Image number : 0")
    change_fig = ChangeFig(len(img_stack),img,img_stack)
    plt.show()
    
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