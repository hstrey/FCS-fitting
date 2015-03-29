# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:57:09 2015

@author: hstrey
"""

import numpy as np
from lmfit import Parameters, minimize, fit_report
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import pickle
import collections

from GaussianModels import g_all,g_all_t, g_all_n,g_all_nt

#defines the location of the data
datadir='../sample data/dilution/'

# load the parameters for each fit from the pickle file
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_all.pkl','r') as paraPickleFile:
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     
for color in parameters:
    experiments=pd.read_table(datadir+'Some'+color+'.txt')
    c=[]
    data_list=[]
    std_list=[]
    for i in range(len(experiments)):
        filename=experiments['filename'][i]
        corrSet=pd.read_csv(datadir+filename+'.csv')
        
        #data set for fitting mean square displacements
        corrData=corrSet[corrSet['delta_t']>=6e-8]
        corrData=corrData[corrData['delta_t']<=0.05]
        
        # get all the parameters from the file
        dataName='mean'+color
        stdName='std'+color
        
        data_list.append(corrData[dataName])
        std_list.append(corrData[stdName])
        c.append("C_"+filename)
        
    data=np.array(data_list)
    std=np.array(std_list)

    t=corrData['delta_t']
    gfit_all=g_all(parameters[color][0],t,c)
    gfit_all_t=g_all_t(parameters[color][1],t,c)
    gfit_all_n=g_all_n(parameters[color][2],t,c)
    gfit_all_nt=g_all_nt(parameters[color][3],t,c)
    
    num_rows=len(experiments)
    
    plt.figure()
    plt.title(color+' combined diluation fits')
    for i,filename in enumerate(experiments['filename']):
        plt.subplot(num_rows,2,i*2+1)
        plt.errorbar(t,data[i],yerr=std[i],fmt='or')
        plt.plot(t,gfit_all[i])
        plt.plot(t,gfit_all_t[i])
        plt.xscale('log')
        plt.ylabel(filename)
        
        plt.subplot(num_rows,2,i*2+2)
        plt.errorbar(t,data[i],yerr=std[i],fmt='or')
        plt.plot(t,gfit_all_n[i])
        plt.plot(t,gfit_all_nt[i])
        plt.xscale('log')
       
