# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:13:04 2018

@author: Jules Scholler
"""

import os

def returnFilesPath(filename, directory='.\\'):
    mylist = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name == filename:
                mylist.append(os.path.join(path, name))
    return mylist