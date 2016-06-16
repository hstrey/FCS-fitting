import numpy as np
from lmfit import Parameter, Parameters
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import time
import sys

from fcsfit.gaussian import modelFCS_t, modelFCS
from fcsfit.realistic_reversed import modelFCS_n, modelFCS_nt, vol1dict, vol2dict
from fcsfit.common import k_real

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
datadir='../data/dilutions/RAW/'
parameter_file='S'

color=raw_input("What color (B or R): ")

experiments=pd.read_table(datadir+color+parameter_file+'.txt')

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

logfile=open(datadir+filename+"_"+color+"_"+parameter_file+'_sinlge.log',"w")
parameterDataFrame=pd.DataFrame({})

# get all the parameters from the file
dataName='mean'+color
stdName='stderr'+color
conc=experiments['conc'+color+'(nM)'][choice]
diffC=experiments['diffcoef'+color][choice]
lambdaem=experiments['lambdaem'+color+'(nm)'][choice]/1000.0
lambdaex=experiments['lambdaex'+color+'(nm)'][choice]/1000.0
a=experiments['fiber'+color+'(microns)'][choice]
print '*** dataset: ',filename,'color: ',color,'conc: ',conc,'diffC: ',diffC
logfile.write('Color: '+color+'\n')

# calculate fit to data and fit to noise
logdata=np.log10(corrData[stdName])
logt=np.log10(corrData['delta_t'])
pf=np.polyfit(logt,logdata,4)
p=np.poly1d(pf)
print pf
fitNoise=10**p(logt)

result_dict['delta_t']=corrData['delta_t']
result_dict['g(t)']=corrData[dataName]
result_dict['std_err']=corrData[stdName]
result_dict['std_err_fit']=fitNoise

# then refit the FCS data using a numerical fit
resultN = modelFCS_n.fit(corrData[dataName],t=corrData['delta_t'],
    C=Parameter(value=5.0, vary=True, min=0.1,max=10.0),
    w0=Parameter(value=0.30069997,vary=False),
    r0=Parameter(value=0.17734422, vary=False),
    n=Parameter(value=1.33, vary=False),
    D=Parameter(value=diffC, vary = False),
    a=Parameter(value=a,vary=False),
    lambdaem=Parameter(value=lambdaem,vary=False),
    lambdaex=Parameter(value=lambdaex,vary=False),
    weights=1./fitNoise)
    
print resultN.fit_report()
logfile.write(resultN.fit_report()+'\n')
v1=vol1dict(resultN.params)
v2=vol2dict(resultN.params)
print "Volume = ",v1*v1/v2

fitFCSN=resultN.best_fit
result_dict['Nfit']=resultN.best_fit
result_dict['Nres']=resultN.residual

results=pd.DataFrame(result_dict)
results.to_csv(datadir+"rev_"+filename+"_"+color+"_"+parameter_file+"_single.csv")

endtime=time.time()

print "Total runtime: ",endtime-starttime
logfile.close()

plt.figure()
plt.errorbar(corrData['delta_t'],corrData[dataName],yerr=fitNoise,fmt="ob")
plt.plot(corrData['delta_t'],fitFCSN,"g",label='Numerical')
plt.xscale('log')
plt.xlabel('delta t in sec')
plt.ylabel('g(t) '+color)

plt.savefig(datadir+filename+"_"+color+"_"+parameter_file+'_single.png', bbox_inches='tight')
plt.savefig(datadir+filename+"_"+color+"_"+parameter_file+'_sinlge.pdf', bbox_inches='tight')

#paraPickleFile.close()
plt.show()

