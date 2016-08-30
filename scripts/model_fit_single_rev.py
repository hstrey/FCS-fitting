import numpy as np
from lmfit import Parameter, Parameters
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import time
import sys

from fcsfit.FCS_Models_reversed import modelFCS_t, modelFCS, modelFCS_nr, modelFCS_ntr, modelFCS_n, modelFCS_nt,vol1, vol2, g_t, g_n, g, g_nt,k_real

def makeResultDataFrame(modelfit,dataset={}):
    params={}
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
datadir='../data/dilutions/RAW/'
parameter_file='S'
color="B"

experiments=pd.read_table(datadir+parameter_file+'.txt')

for i in range(len(experiments)):
    print i,experiments['filename'][i]

choice=input("Please choose one dataset: ")
filename=experiments['filename'][choice]

print filename

result_dict={}
corrSet=pd.read_csv(datadir+filename+'.csv')

#data set for fitting mean square displacements
low_cutoff=raw_input("low cutoff in s (1e-7): ")
if low_cutoff=="":
    low_cutoff=1e-7
else:
    low_cutoff=float(low_cutoff)

high_cutoff=raw_input("high cutoff in s (1e-2): ")
if high_cutoff=="":
    high_cutoff=1e-2
else:
    high_cutoff=float(high_cutoff)
    
corrData=corrSet[corrSet['delta_t']>=low_cutoff]
corrData=corrData[corrData['delta_t']<=high_cutoff]

colorChoice=input("Choose color Blue=0, Red=1: ")
color=['B','R'][colorChoice]

logfile=open(datadir+filename+"_"+color+"_"+parameter_file+'.log',"w")
parameterDataFrame=pd.DataFrame({})

#paraPickleFile=open(datadir+parameter_file+color+'.pkl',"w")
   
# get all the parameters from the file
dataName='mean'+color
stdName='std'+color
conc=experiments['conc'+color+'(nM)'][choice]
diffC=experiments['diffcoef'+color][choice]
lambdaem=experiments['lambdaem'+color+'(nm)'][choice]/1000.0
lambdaex=experiments['lambdaex'+color+'(nm)'][choice]/1000.0
a=experiments['fiber'+color+'(microns)'][choice]
print '*** dataset: ',filename,'color: ',color,'conc: ',conc,'diffC: ',diffC
logfile.write('Color: '+color+'\n')

# calculate fit to data and fit to noise
logdata=np.log10(corrData[stdName]/np.sqrt(20))
logt=np.log10(corrData['delta_t'])
pf=np.polyfit(logt,logdata,4)
p=np.poly1d(pf)
print pf
fitNoise=10**p(logt)

result_dict['delta_t']=corrData['delta_t']
result_dict['g(t)']=corrData[dataName]
result_dict['std_err']=corrData[stdName]
result_dict['std_err_fit']=fitNoise

resultG = modelFCS.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
    wxy=0.3,
    wz=1.0,
    D=Parameter(value=diffC, vary = False),
    weights=1./fitNoise)

print resultG.fit_report()
logfile.write(resultG.fit_report()+'\n')
fitFCS=resultG.best_fit

result_dict['Gfit']=resultG.best_fit

datasetG=makeResultDataFrame(resultG,{'fit':'fcs','v':resultG.values['wxy']**2*resultG.values['wz']*m.pi**1.5,'chisq':resultG.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetG,ignore_index=True)

#pickle.dump(resultG,paraPickleFile)

# then refit the FCS data with triplet correction
resultS = modelFCS_t.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
    wxy=Parameter(value=resultG.values['wxy'],vary=True),
    wz=Parameter(value=resultG.values['wz'],vary=True),
    F=Parameter(value=0.02,min=0.0,max=0.5,vary=True),
    tf=Parameter(5e-7,vary=True, min=1e-9, max=1e-4),
    D=Parameter(value=diffC, vary = False),
    weights=1./fitNoise)
    
print resultS.fit_report()
logfile.write(resultS.fit_report()+'\n')
   
fitFCST=resultS.best_fit
result_dict['Tfit']=resultS.best_fit
result_dict['Tres']=resultS.residual

datasetS=makeResultDataFrame(resultS,{'fit':'fcs_t','v':resultG.values['wxy']**2*resultG.values['wz']*m.pi**1.5,'chisq':resultS.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetS,ignore_index=True)

#pickle.dump(resultS,paraPickleFile)

# then refit the FCS data using a numerical fit
resultN = modelFCS_n.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
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

fitFCSN=resultN.best_fit
result_dict['Nfit']=resultN.best_fit
result_dict['Nres']=resultN.residual

datasetN=makeResultDataFrame(resultN,{'fit':'fcs_n','v':v1*v1/v2,'chisq':resultN.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetN,ignore_index=True)

#pickle.dump(resultN,paraPickleFile)

resultNT = modelFCS_nt.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
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

fitFCSNT=resultNT.best_fit
result_dict['NTfit']=resultNT.best_fit
result_dict['NTres']=resultNT.residual

datasetNT=makeResultDataFrame(resultNT,{'fit':'fcs_nt','v':v1*v1/v2,'chisq':resultNT.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetNT,ignore_index=True)

#pickle.dump(resultNT,paraPickleFile)
# then refit the FCS data using a numerical fit
resultNR = modelFCS_nr.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
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
    
print resultNR.fit_report()
logfile.write(resultNR.fit_report()+'\n')
para_dict=resultNR.values
para_dict['k']=k_real
v1=vol1(**para_dict)
v2=vol2(**para_dict)
print "Volume = ",v1*v1/v2

fitFCSNR=resultNR.best_fit
result_dict['NRfit']=resultNR.best_fit
result_dict['NRres']=resultNR.residual

datasetNR=makeResultDataFrame(resultNR,{'fit':'fcs_nr','v':v1*v1/v2,'chisq':resultNR.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetNR,ignore_index=True)

#pickle.dump(resultN,paraPickleFile)

resultNTR = modelFCS_ntr.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=conc, vary=True),
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
    
print resultNTR.fit_report()
logfile.write(resultNTR.fit_report()+'\n')
para_dict=resultNTR.values
para_dict['k']=k_real
v1=vol1(**para_dict)
v2=vol2(**para_dict)
print "Volume = ",v1*v1/v2

fitFCSNTR=resultNTR.best_fit
result_dict['NTRfit']=resultNTR.best_fit
result_dict['NTRres']=resultNTR.residual

datasetNTR=makeResultDataFrame(resultNTR,{'fit':'fcs_ntr','v':v1*v1/v2,'chisq':resultNTR.chisqr})
parameterDataFrame=parameterDataFrame.append(datasetNTR,ignore_index=True)

results=pd.DataFrame(result_dict)
results.to_csv(datadir+"rev_"+filename+"_"+color+"_"+parameter_file+".csv")

parameterDataFrame.to_csv(datadir+"para_rev_"+filename+"_"+color+"_"+parameter_file+".csv")

endtime=time.time()

print "Total runtime: ",endtime-starttime

plt.figure(figsize=(10, 10))
plt.subplot(2,3,1)
plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
plt.plot(corrData['delta_t'],fitFCS,"g",label='3d-Gaussian')
plt.plot(corrData['delta_t'],fitFCST,"r",label='+Triplet')
plt.xscale('log')
plt.xlabel('delta t in sec')
plt.ylabel('g(t) '+color)
plt.legend(frameon=False)

plt.subplot(2,3,2)
plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
plt.plot(corrData['delta_t'],fitFCSN,"g",label='Numerical')
plt.plot(corrData['delta_t'],fitFCSNT,"r",label='+Triplet')
plt.xscale('log')
plt.xlabel('delta t in sec')
plt.ylabel('g(t) '+color)
plt.legend(frameon=False)

plt.subplot(2,3,3)
plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
plt.plot(corrData['delta_t'],fitFCSNR,"g",label='Numerical')
plt.plot(corrData['delta_t'],fitFCSNTR,"r",label='+Triplet')
plt.xscale('log')
plt.xlabel('delta t in sec')
plt.ylabel('g(t) '+color)
plt.legend(frameon=False)

plt.subplot(2,3,4)
plt.semilogx(corrData['delta_t'],resultG.residual,"g")
plt.semilogx(corrData['delta_t'],resultS.residual,"r")
plt.xlabel('delta t in sec')
plt.ylabel('residuals')
  
plt.subplot(2,3,5)
plt.semilogx(corrData['delta_t'],resultN.residual,"g")
plt.semilogx(corrData['delta_t'],resultNT.residual,"r")
plt.xlabel('delta t in sec')
plt.ylabel('residuals')

plt.subplot(2,3,6)
plt.semilogx(corrData['delta_t'],resultNR.residual,"g")
plt.semilogx(corrData['delta_t'],resultNTR.residual,"r")
plt.xlabel('delta t in sec')
plt.ylabel('residuals')

plt.savefig(datadir+filename+"_"+color+"_"+parameter_file+'.png', bbox_inches='tight')
plt.savefig(datadir+filename+"_"+color+"_"+parameter_file+'.pdf', bbox_inches='tight')

logfile.close()
#paraPickleFile.close()
plt.show()

