# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:13:04 2018

@author: Jules Scholler
"""

import os
import h5py
import numpy as np

def return_files(filename=None, directory='.\\', extension=None, method='keyword'):
    mylist = []
    if method == 'filename' and extension == None:
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if name == filename:
                    mylist.append(os.path.join(path, name))
    if method == 'filename' and extension != None:
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if name == filename + extension:
                    mylist.append(os.path.join(path, name))
    elif method == 'keyword' and extension != None:
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if filename in name and name[-len(extension):] == extension:
                    mylist.append(os.path.join(path, name))
    elif method == 'keyword' and extension == None:
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if filename in name:
                    mylist.append(os.path.join(path, name))
    elif method == 'extension':
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if name[-len(extension):] == extension:
                    mylist.append(os.path.join(path, name))
    return mylist

def loadmatv7(pathname):
    f = h5py.File(pathname)
    data = {}
    for k, v in f.items():
        data[k] = np.array(v)
    return data