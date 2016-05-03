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
        
# here we should ask the user for all the parameters of the datasets
# maybe ask: do you want to create All.txt

for filestart in groups:
    corr_arrayb=[]
    corr_arrayr=[]
    
    for file in groups[filestart]:
        para, corrfcts, intensity = read_corrfile(datadir+file)
        corrfcts=pd.DataFrame(np.array(corrfcts[:-1]),columns=['delta_t','corrAB','corrBA','corrCD','corrDC'])
        corr_arrayb.extend([corrfcts['corrAB'],corrfcts['corrBA']])
        corr_arrayr.extend([corrfcts['corrCD'],corrfcts['corrDC']])

    corr_arrayb=np.array(corr_arrayb)
    corr_arrayr=np.array(corr_arrayr)

    corr_meanB = np.mean(corr_arrayb,axis=0)
    corr_meanR = np.mean(corr_arrayr,axis=0)

    corr_stdB = np.std(corr_arrayb,axis=0)
    corr_stdR = np.std(corr_arrayr,axis=0)

    corr_stderrB = corr_stdB/np.sqrt(corr_arrayb.shape[0]) # calculate stderr
    corr_stderrR = corr_stdR/np.sqrt(corr_arrayr.shape[0]) # calculate stderr

    d={'delta_t':corrfcts['delta_t'], 'meanB':corr_meanB, 'stdB':corr_stdB,'stderrB':corr_stderrB,'meanR':corr_meanR, 'stdR':corr_stdR,'stderrR':corr_stderrR}
    df=pd.DataFrame(d)
    print df
    df.to_csv(datadir+filestart+'.csv',index=False)
