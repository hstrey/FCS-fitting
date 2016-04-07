# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:19:30 2015

@author: hstrey
"""
import os
import collections
import pandas as pd
import numpy as np

#defines the location of the data
datadir='../data/dilutions/RAW/'

filelist=os.listdir(datadir)

listB=[]
listR=[]

for file in filelist:
    if file.find('.csv')==-1:
        continue
    print file
    if file.startswith('BO'):
        listB.append(file)
    if file.startswith('OR'):
        listR.append(file)

listR.sort()
listB.sort()

print listR,listB

# here we should ask the user for all the parameters of the datasets
# maybe ask: do you want to create All.txt

for fileB, fileR in zip(listB, listR):
    print fileB,fileR
