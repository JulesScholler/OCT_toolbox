# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:13:04 2018

@author: Jules Scholler
"""

import os
import h5py
import numpy as np

def returnFilesPath(filename, directory='.\\'):
    mylist = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name == filename:
                mylist.append(os.path.join(path, name))
    return mylist

def loadmatv7(pathname):
    f = h5py.File(pathname)
    data = {}
    for k, v in f.items():
        data[k] = np.array(v)
    return data