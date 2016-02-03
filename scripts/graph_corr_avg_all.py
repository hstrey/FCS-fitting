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

from fcsfit.FCS_Models import g_all,g_all_t, g_all_n,g_all_nt,g_all_nr,g_all_ntr,vol1dict, vol2dict

def makeResultDataFrame(params,dataset={}):
    
    parnames = sorted(params)
    

    for name in parnames:
        par=params[name]
        if par.value is not None:
            dataset[name]=par.value
        if par.vary:
            if par.stderr is not None:
                dataset[name+'_stderr']=par.stderr
                
    return pd.DataFrame(dataset,index=[0])

#defines the location of the data
datadir='../data/dilutions/SOME/'
parafile="B4R4"

# load the parameters for each fit from the pickle file
# there are 6 parameter objects per color 3dG, 3dGt, n, nt, nr, ntr
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_'+parafile+'.pkl',"r") as paraPickleFile:
    for i in range(6):
        parameters['B'].append(pickle.load(paraPickleFile))
    for i in range(6):
        parameters['R'].append(pickle.load(paraPickleFile))
     
for color in parameters:
    experiments=pd.read_table(datadir+'Some'+color+'_'+parafile+'_L.txt')
    c=[]
    data_list=[]
    std_list=[]
    for i in range(len(experiments)):
        filename=experiments['filename'][i]
        corrSet=pd.read_csv(datadir+filename+'.csv')
        
        #data set for fitting mean square displacements
        corrData=corrSet[corrSet['delta_t']>=1e-7]
        corrData=corrData[corrData['delta_t']<=0.05]
        
        # get all the parameters from the file
        dataName='mean'+color
        stdName='std'+color
        
        data_list.append(corrData[dataName])
        std_list.append(corrData[stdName])
        c.append("C_"+filename)
        
    data=np.array(data_list)
    std=np.array(std_list)/np.sqrt(20)

    t=corrData['delta_t']
    gfit_all=g_all(parameters[color][0],t,c)
    gfit_all_t=g_all_t(parameters[color][1],t,c)
    gfit_all_n=g_all_n(parameters[color][2],t,c)
    gfit_all_nr=g_all_nr(parameters[color][3],t,c)
    gfit_all_nt=g_all_nt(parameters[color][4],t,c)
    gfit_all_ntr=g_all_ntr(parameters[color][5],t,c)
    
    gfit_all_res=(gfit_all-data)/std
    gfit_all_t_res=(gfit_all_t-data)/std
    gfit_all_n_res=(gfit_all_n-data)/std
    gfit_all_nt_res=(gfit_all_nt-data)/std
    gfit_all_nr_res=(gfit_all_nr-data)/std
    gfit_all_ntr_res=(gfit_all_ntr-data)/std
    
    datadict={}
    for i,filename in enumerate(experiments['filename']):
        datadict['g_all_'+filename]=gfit_all[i]
        datadict['g_all_t_'+filename]=gfit_all_t[i]
        datadict['g_all_n_'+filename]=gfit_all_n[i]
        datadict['g_all_nr_'+filename]=gfit_all_nr[i]
        datadict['g_all_nt_'+filename]=gfit_all_nt[i]
        datadict['g_all_ntr_'+filename]=gfit_all_ntr[i]

    datadict['g_all_dt']=t
    for i,filename in enumerate(experiments['filename']):
        datadict['g_all_res']=gfit_all_res[i]
        datadict['g_all_t_res']=gfit_all_t_res[i]
        datadict['g_all_n_res']=gfit_all_n_res[i]
        datadict['g_all_nt_res']=gfit_all_nt_res[i]
        datadict['g_all_nr_res']=gfit_all_nr_res[i]
        datadict['g_all_ntr_res']=gfit_all_ntr_res[i]
    
    Np=3
    Ntp=5
    print "reduced Chi square", sum(gfit_all_res.flatten()**2)/(len(gfit_all_res.flatten())-Np)
    print "reduced Chi square", sum(gfit_all_t_res.flatten()**2)/(len(gfit_all_t_res.flatten())-Ntp)
    print "reduced Chi square", sum(gfit_all_n_res.flatten()**2)/(len(gfit_all_n_res.flatten())-Np)
    print "reduced Chi square", sum(gfit_all_nt_res.flatten()**2)/(len(gfit_all_nt_res.flatten())-Ntp)
    print "reduced Chi square", sum(gfit_all_nr_res.flatten()**2)/(len(gfit_all_nr_res.flatten())-Np)
    print "reduced Chi square", sum(gfit_all_ntr_res.flatten()**2)/(len(gfit_all_ntr_res.flatten())-Ntp)
    
    dataFits=pd.DataFrame(datadict)
    dataFits.to_csv(datadir+'all_fits_'+parafile+".csv")

    num_rows=len(experiments)
    
    plt.figure()
    plt.title(color+' combined diluation fits 3d Gaussian')
    for i,filename in enumerate(experiments['filename']):
        plt.subplot(num_rows,1,i+1)
        plt.errorbar(t,data[i],yerr=std[i],fmt='or')
        plt.plot(t,gfit_all[i])
        plt.plot(t,gfit_all_t[i])
        plt.xscale('log')
        plt.ylabel(filename)
        
    plt.figure()
    plt.title(color+' combined diluation fits numerical')
    for i,filename in enumerate(experiments['filename']):
        plt.subplot(num_rows,1,i+1)
        plt.errorbar(t,data[i],yerr=std[i],fmt='or')
        plt.plot(t,gfit_all_n[i])
        plt.plot(t,gfit_all_nt[i])
        plt.xscale('log')
        plt.ylabel(filename)
        
    plt.figure()
    plt.title(color+' combined diluation fits numerical real')
    for i,filename in enumerate(experiments['filename']):
        plt.subplot(num_rows,1,i+1)
        plt.errorbar(t,data[i],yerr=std[i],fmt='or')
        plt.plot(t,gfit_all_nr[i])
        plt.plot(t,gfit_all_ntr[i])
        plt.xscale('log')
        plt.ylabel(filename)


