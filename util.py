# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:13:04 2018

@author: Jules Scholler
"""

import os
import h5py
import numpy as np

def return_files(filename=None, directory='.\\', extension=None, method='keyword'):
    ''' this function returns a list of file given some information (filenam, keyword or extension) or a combinaison
        of information. It will recursively look into the given folders and all subfolders. '''
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
    ''' This function load .mat files recorded with the newest format (v7.3) as a numpy array. '''
    f = h5py.File(pathname)
    data = {}
    for k, v in f.items():
        data[k] = np.array(v)
    return data

def move_files(directory_i, directory_f, filename, extension):
    ''' This function moves files from a given folder to another. '''
    paths = []
    names = []
    for path, subdirs, files in os.walk(directory_i):
        for name in files:
            if filename in name and name[-len(extension):] == extension:
                paths.append(path)
                names.append(name)
    
    for i, pathname in enumerate(paths):
        os.rename(pathname + '\\' + names[i], directory_f + names[i])

def rename_files(directory, filename, extension, index_shift):
    ''' This function renames files with digits with a fixed increment. '''
    paths = []
    names = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if filename in name and name[-len(extension):] == extension:
                paths.append(path)
                names.append(name)

    for i, pathname in enumerate(paths):
        number = [s for s in names[i].replace('_',' ').replace('.',' ').split() if s.isdigit()][0]
        new_number = str(int(number) + index_shift)
        new_name = names[i].replace(number, new_number)
        os.rename(pathname + '\\' + names[i], pathname + '\\' + new_name)