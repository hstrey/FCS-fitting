# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:57:09 2015

@author: hstrey
"""

import numpy as np
from lmfit import Parameters, Parameter
import pandas as pd
import pickle
import collections

from fcsfit.gaussian import modelFCS_t, modelFCS
from fcsfit.realistic_reversed import modelFCS_nr, modelFCS_ntr, modelFCS_n, modelFCS_nt

def makeResultDataFrame(modelfit,dataset={}):
    if isinstance(modelfit, Parameters):
        params = modelfit
    if hasattr(modelfit, 'params'):
        params = modelfit.params

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
datadir_all='../data/dilutions/'
parafile="B4R4"

# load the parameters for each fit from the pickle file
# there are 6 parameter objects per color 3dG, 3dGt, n, nt, nr, ntr
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_rev2.pkl',"r") as paraPickleFile:
    for i in range(6):
        parameters['B'].append(pickle.load(paraPickleFile))
    for i in range(6):
        parameters['R'].append(pickle.load(paraPickleFile))

experiments=pd.read_table(datadir_all+parafile+'.txt')
logfile=open(datadir_all+'fit_dilutions_rev2.log',"w")
parameterDataFrame=pd.DataFrame({})
datadict={}

for color in parameters:
    for i in range(len(experiments)):
        filename=experiments['filename'][i]
        corrSet=pd.read_csv(datadir_all+filename+'.csv')
        
        #data set for fitting mean square displacements
        corrData=corrSet[corrSet['delta_t']>=1e-7]
        corrData=corrData[corrData['delta_t']<=0.02]
        
        # get all the parameters from the file
        dataName='mean'+color
        stdName='std'+color
                
        data=np.array(corrData[dataName])
        std=np.array(corrData[stdName])/np.sqrt(20)
        t=corrData['delta_t']
        
        conc=experiments['conc'+color+'(nM)'][i]
        diffC=experiments['diffcoef'+color][i]
        lambdaem=experiments['lambdaem'+color+'(nm)'][i]/1000.0
        lambdaex=experiments['lambdaex'+color+'(nm)'][i]/1000.0
        a=experiments['fiber'+color+'(microns)'][i]
        print '*** dataset: ',filename,'color: ',color,'conc: ',conc,'diffC: ',diffC
        logfile.write('Color: '+color+'\n')
        
        # calculate fit to data and fit to noise
        logdata=np.log10(std)
        logt=np.log10(corrData['delta_t'])
        pf=np.polyfit(logt,logdata,4)
        p=np.poly1d(pf)
        fitNoise=10**p(logt)
        
        # save data, stderr and fitnoise just in case
        datadict['data_'+filename+'_'+color]=data
        datadict['std_'+filename+'_'+color]=std
        datadict['fitstd_'+filename+'_'+color]=fitNoise
        
        fit=parameters[color][0] # Gaussian fit pickle

        resultG = modelFCS.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            wxy=Parameter(value=fit['wxy'].value,vary=False),
            wz=Parameter(value=fit['wz'].value,vary=False),
            D=Parameter(value=diffC, vary = False),
            weights=1./fitNoise)
        
        print resultG.fit_report()
        logfile.write(resultG.fit_report()+'\n')
        datasetG=makeResultDataFrame(resultG,{'fit':'fcs','file':filename,'color':color,'chisq':resultG.chisqr,'redchi':resultG.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetG,ignore_index=True)
        
        datadict['fitdata_fcs_'+filename+'_'+color]=resultG.best_fit

        fit=parameters[color][1] # Gaussian fit triplet pickle

        resultS = modelFCS_t.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            wxy=Parameter(value=fit['wxy'].value,vary=False),
            wz=Parameter(value=fit['wz'].value,vary=False),
            F=Parameter(value=fit['F'].value,min=0.0,max=0.5,vary=True),
            tf=Parameter(value=fit['tf'].value,vary=True, min=1e-9, max=1e-4),
            D=Parameter(value=diffC, vary = False),
            weights=1./fitNoise)
            
        print resultS.fit_report()
        logfile.write(resultS.fit_report()+'\n')
        datasetS=makeResultDataFrame(resultS,{'fit':'fcs_t','file':filename,'color':color,'chisq':resultS.chisqr,'redchi':resultS.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetS,ignore_index=True)

        datadict['fitdata_fcst_'+filename+'_'+color]=resultS.best_fit

        fit=parameters[color][2] # Numerical fit pickle
        
        resultN = modelFCS_n.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            w0=Parameter(value=fit['w0'].value,vary=False),
            r0=Parameter(value=fit['r0'].value,vary=False),
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultN.fit_report()
        logfile.write(resultN.fit_report()+'\n')
        datasetN=makeResultDataFrame(resultN,{'fit':'fcs_n','file':filename,'color':color,'chisq':resultN.chisqr,'redchi':resultN.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetN,ignore_index=True)
        
        datadict['fitdata_n_'+filename+'_'+color]=resultN.best_fit

        fit=parameters[color][4] # Numerical fit triplet pickle
        
        resultN = modelFCS_nt.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            w0=Parameter(value=fit['w0'].value,vary=False),
            r0=Parameter(value=fit['r0'].value,vary=False),
            F=Parameter(value=fit['F'].value,min=0.0,max=0.5,vary=True),
            tf=Parameter(value=fit['tf'].value,vary=True, min=1e-9, max=1e-4),
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultN.fit_report()
        logfile.write(resultN.fit_report()+'\n')
        datasetN=makeResultDataFrame(resultN,{'fit':'fcs_nt','file':filename,'color':color,'chisq':resultN.chisqr,'redchi':resultN.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetN,ignore_index=True)
        
        datadict['fitdata_nt_'+filename+'_'+color]=resultN.best_fit

        fit=parameters[color][3] # NumericalR fit pickle
        
        resultN = modelFCS_nr.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            w0=Parameter(value=fit['w0'].value,vary=False),
            r0=Parameter(value=fit['r0'].value,vary=False),
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultN.fit_report()
        logfile.write(resultN.fit_report()+'\n')
        datasetN=makeResultDataFrame(resultN,{'fit':'fcs_nr','file':filename,'color':color,'chisq':resultN.chisqr,'redchi':resultN.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetN,ignore_index=True)
        
        datadict['fitdata_nr_'+filename+'_'+color]=resultN.best_fit

        fit=parameters[color][5] # NumericalR fit triplet pickle
        
        resultN = modelFCS_ntr.fit(data,t=t,
            C=Parameter(value=conc*fit['slope'].value, vary=True),
            w0=Parameter(value=fit['w0'].value,vary=False),
            r0=Parameter(value=fit['r0'].value,vary=False),
            F=Parameter(value=fit['F'].value,min=0.0,max=0.5,vary=True),
            tf=Parameter(value=fit['tf'].value,vary=True, min=1e-9, max=1e-4),
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultN.fit_report()
        logfile.write(resultN.fit_report()+'\n')
        datasetN=makeResultDataFrame(resultN,{'fit':'fcs_ntr','file':filename,'color':color,'chisq':resultN.chisqr,'redchi':resultN.redchi})
        parameterDataFrame=parameterDataFrame.append(datasetN,ignore_index=True)
        datadict['fitdata_ntr_'+filename+'_'+color]=resultN.best_fit

    # save the delta time for each color just in case
    datadict['t'+color]=t    

parameterDataFrame.to_csv(datadir_all+"fit_dilutions_rev2.csv")
dataFits=pd.DataFrame(datadict)
dataFits.to_csv(datadir_all+'fit_dilutions_plots_rev2.csv')
