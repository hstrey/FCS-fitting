import numpy as np
from lmfit import Parameter, Parameters
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import time

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

starttime=time.time()
#defines the location of the data
datadir='../data/dilutions/'

data3dGaussian=pd.DataFrame({})
data3dGaussianTriplet=pd.DataFrame({})
dataNumerical=pd.DataFrame({})
dataNumericalTriplet=pd.DataFrame({})

experiments=pd.read_table(datadir+'BSRS.txt')

for i in range(len(experiments)):
    filename=experiments['filename'][i]
    logfile=open(datadir+filename+'BSRS.log',"w")
    corrSet=pd.read_csv(datadir+filename+'.csv')
    
    #data set for fitting mean square displacements
    corrData=corrSet[corrSet['delta_t']>=1e-7]
    corrData=corrData[corrData['delta_t']<=1e-2]
    
    for color in ['B','R']:
        
        # get all the parameters from the file
        dataName='mean'+color
        stdName='std'+color
        conc=experiments['conc'+color+'(nM)'][i]
        diffC=experiments['diffcoef'+color][i]
        lambdaem=experiments['lambdaem'+color+'(nm)'][i]/1000.0
        lambdaex=experiments['lambdaex'+color+'(nm)'][i]/1000.0
        a=experiments['fiber'+color+'(microns)'][i]
        print '*** dataset: ',filename,'color: ',color,'conc: ',conc,'diffC: ',diffC
        logfile.write('Color: '+color+'\n')
        
        # calculate fit to data and fit to noise
        logdata=np.log10(corrData[stdName])
        logt=np.log10(corrData['delta_t'])
        pf=np.polyfit(logt,logdata,4)
        p=np.poly1d(pf)
        print pf
        fitNoise=10**p(logt)

        resultG = modelFCS.fit(corrData[dataName],t=corrData['delta_t'],
            C=Parameter(value=conc, vary=False),
            wxy=0.3,
            wz=1.0,
            D=Parameter(value=diffC, vary = False),
            weights=1./fitNoise)
        
        print resultG.fit_report()
        logfile.write(resultG.fit_report()+'\n')
        fitFCS=g(t=corrData['delta_t'],**resultG.values)
        
        #create a new row of data and append to result
        datasetG=makeResultDataFrame(resultG,{'dataset':filename,'color':color,'v':resultG.values['wxy']**2*resultG.values['wz']*m.pi**1.5})
        data3dGaussian=data3dGaussian.append(datasetG,ignore_index=True)
        
        # then refit the FCS data using the noise as sigma
        resultS = modelFCS_t.fit(corrData[dataName],t=corrData['delta_t'],
            C=Parameter(value=conc, vary=False),
            wxy=0.3,
            wz=1.0,
            F=Parameter(value=0.01,min=0.0,max=0.5,vary=True),
            tf=Parameter(5e-7,vary=True, min=1e-9, max=1e-4),
            D=Parameter(value=diffC, vary = False),
            weights=1./fitNoise)
            
        print resultS.fit_report()
        logfile.write(resultS.fit_report()+'\n')
       
        fitFCST=g_t(t=corrData['delta_t'],**resultS.values)
        
        #create a new row of data and append to result
        datasetT=makeResultDataFrame(resultS,{'dataset':filename,'color':color,'v':resultS.values['wxy']**2*resultS.values['wz']*m.pi**1.5})
        data3dGaussianTriplet=data3dGaussianTriplet.append(datasetT,ignore_index=True)

        # then refit the FCS data using a numerical fit
        resultN = modelFCS_nr.fit(corrData[dataName],t=corrData['delta_t'],
            C=Parameter(value=conc, vary=False),
            w0=0.25,
#            F=Parameter(value=0.1,min=0.0,max=0.5,vary=True),
#            tf=1e-7,
            r0=0.16,
#            F=Parameter(value=resultS.values['F'],min=0.0,max=0.5,vary=True),
#            tf=resultS.values['tf'],
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultN.fit_report()
        logfile.write(resultN.fit_report()+'\n')
        v1=vol1(**resultN.values)
        v2=vol2(**resultN.values)
        print "Volume = ",v1*v1/v2
        
        fitFCSN=g_n(t=corrData['delta_t'],**resultN.values)
        
        datasetN=makeResultDataFrame(resultN,{'dataset':filename,'color':color,'v':v1*v1/v2})
        dataNumerical=dataNumerical.append(datasetN,ignore_index=True)

        resultNT = modelFCS_ntr.fit(corrData[dataName],t=corrData['delta_t'],
            C=Parameter(value=conc, vary=False),
            w0=Parameter(value=resultN.values['w0'],vary=True),
#            F=Parameter(value=0.1,min=0.0,max=0.5,vary=True),
#            tf=1e-7,
            r0=Parameter(value=resultN.values['r0'],vary=True),
            F=Parameter(value=0.02,min=0.0,max=0.5,vary=True),
            tf=Parameter(value=5e-7,vary=True, min=1e-9, max=1e-4),
            n=Parameter(value=1.33, vary=False),
            D=Parameter(value=diffC, vary = False),
            a=Parameter(value=a,vary=False),
            lambdaem=Parameter(value=lambdaem,vary=False),
            lambdaex=Parameter(value=lambdaex,vary=False),
            weights=1./fitNoise)
            
        print resultNT.fit_report()
        logfile.write(resultNT.fit_report()+'\n')
        v1=vol1(**resultNT.values)
        v2=vol2(**resultNT.values)
        print "Volume = ",v1*v1/v2
        
        fitFCSNT=g_nt(t=corrData['delta_t'],**resultNT.values)
        
        datasetNT=makeResultDataFrame(resultNT,{'dataset':filename,'color':color,'v':v1*v1/v2})
        dataNumericalTriplet=dataNumericalTriplet.append(datasetNT,ignore_index=True)

        plt.figure(figsize=(10, 10))
        plt.subplot(3,2,1)
        plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
        plt.plot(corrData['delta_t'],fitFCS,"g",label='3d-Gaussian')
        plt.plot(corrData['delta_t'],fitFCST,"r",label='+Triplet')
        plt.xscale('log')
        plt.xlabel('delta t in sec')
        plt.ylabel('g(t) '+color)
        plt.legend(frameon=False)
        
        plt.subplot(3,2,2)
        plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
        plt.plot(corrData['delta_t'],fitFCSN,"g",label='Numerical')
        plt.plot(corrData['delta_t'],fitFCSNT,"r",label='+Triplet')
        plt.xscale('log')
        plt.xlabel('delta t in sec')
        plt.ylabel('g(t) '+color)
        plt.legend(frameon=False)
        
        plt.subplot(3,2,3)
        plt.semilogx(corrData['delta_t'],resultG.residual,"g")
        plt.semilogx(corrData['delta_t'],resultS.residual,"r")
        plt.xlabel('delta t in sec')
        plt.ylabel('residuals')
      
        plt.subplot(3,2,4)
        plt.semilogx(corrData['delta_t'],resultN.residual,"g")
        plt.semilogx(corrData['delta_t'],resultNT.residual,"r")
        plt.xlabel('delta t in sec')
        plt.ylabel('residuals')

        plt.subplot(3,2,5)
        plt.loglog(corrData['delta_t'],corrData[stdName],"ob")
        plt.loglog(corrData['delta_t'],fitNoise)
        plt.ylabel('sigma')
        plt.xlabel('delta t in sec')
        
        plt.subplot(3,2,6)
        plt.semilogx(corrData['delta_t'],corrData[stdName],"ob")
        plt.semilogx(corrData['delta_t'],fitNoise)
        plt.ylabel('sigma')
        plt.xlabel('delta t in sec')
        plt.savefig(datadir+filename+color+'BSRS.png', bbox_inches='tight')
        plt.savefig(datadir+filename+color+'BSRS.pdf', bbox_inches='tight')
    logfile.close()
plt.show()

data3dGaussian.to_csv(datadir+'gaussian_BSRS.csv')
data3dGaussianTriplet.to_csv(datadir+'gaussian_triplet_BSRS.csv')
dataNumerical.to_csv(datadir+'Numerical_BSRS.csv')
dataNumericalTriplet.to_csv(datadir+'NumericalTriplet_BSRS.csv')

endtime=time.time()

print "Total runtime: ",endtime-starttime
