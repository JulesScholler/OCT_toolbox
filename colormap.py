# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:05:01 2018

@author: Jules Scholler
"""

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