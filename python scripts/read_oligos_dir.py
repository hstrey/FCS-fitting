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
datadir='../sample data/oligo/'

def read_corrfile(file):
    with open(file,'r') as f:
    
        for line in f:
            if line.startswith('[Para'):
                break
        parameters=[]
        for line in f:
            if line.startswith('[Corr'):
                break
            parameters.append(line)
        corrfcts=[]
        for line in f:
            if line.startswith('[Raw'):
                break
            corrfcts.append([float(a) for a in line.split()])
        for line in f:
            if line.startswith('[IntensityHistory'):
                break
        intensity=[]
        for line in f:
            if line.startswith('[Overflow'):
                break
            if line.startswith('Trace'):
                trace=line
            else:
                intensity.append([float(a) for a in line.split()])
    return parameters,corrfcts,intensity

filelist=os.listdir(datadir)

groups=collections.defaultdict(list)

for file in filelist:
    if file.find('_')==-1:
        continue
    filestart=file[:file.find('_')]
    if file.endswith('.sin') or file.endswith('.SIN'):
        groups[filestart].append(file)

for filestart in groups:
    corr_arrayb=[]
    corr_arrayr=[]
    corr_arraybr=[]
    
    for file in groups[filestart]:
        para, corrfcts, intensity = read_corrfile(datadir+file)
        corrfcts=pd.DataFrame(np.array(corrfcts[:-1]),columns=['delta_t','corrAB','corrCD','corrAD','corrBC'])
        corr_arrayb.append(corrfcts['corrAB'])
        corr_arrayr.append(corrfcts['corrCD'])
        corr_arraybr.extend([corrfcts['corrAD'],corrfcts['corrBC']])
            
    corr_meanb = np.mean(corr_arrayb,axis=0)
    corr_meanr = np.mean(corr_arrayr,axis=0)
    corr_meanbr = np.mean(corr_arraybr,axis=0)
    corr_stdb = np.std(corr_arrayb,axis=0)
    corr_stdr = np.std(corr_arrayr,axis=0)
    corr_stdbr = np.std(corr_arraybr,axis=0)
    d={'delta_t':corrfcts['delta_t'], 'meanB':corr_meanb, 'stdB':corr_stdb,'meanR':corr_meanr, 'stdR':corr_stdr,'meanBR':corr_meanbr, 'stdBR':corr_stdbr}
    df=pd.DataFrame(d)
    print df
    df.to_csv(datadir+filestart+'.csv',index=False)
