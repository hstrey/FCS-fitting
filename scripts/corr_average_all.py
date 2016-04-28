import numpy as np
from lmfit import Parameters, minimize, fit_report
import pandas as pd
import math as m
import pickle
import copy

from fcsfit.gaussian import g_all,g_all_t
from fcsfit.realistic_reversed import g_all_n,g_all_nt,g_all_nr,g_all_ntr,vol1dict, vol2dict,k_real

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
datadir_all='../data/dilutions/RAW/'
parafile="S"

data3dG=pd.read_csv(datadir_all+'gaussian_'+parafile+'.csv')
data3dGT=pd.read_csv(datadir_all+'gaussian_triplet_'+parafile+'.csv')
dataN=pd.read_csv(datadir_all+'Numerical_'+parafile+'.csv')
dataNT=pd.read_csv(datadir_all+'NumericalTriplet_'+parafile+'.csv')

paraPickleFile=open(datadir+'corr_average_all_final.pkl',"w")

for color in ['B','R']:
    data_list=[]
    std_list=[]
    c=[]
    b=Parameters()
    experiments=pd.read_table(datadir+color+parafile+"_SOME.txt")
    logfile=open(datadir+color+'_'+parafile+'_final.log',"w")
    for i in range(len(experiments)):
        filename=experiments['filename'][i]
        corrSet=pd.read_csv(datadir_all+filename+'.csv')
        
        #data set for fitting mean square displacements
        corrData=corrSet[corrSet['delta_t']>=1e-7]
        corrData=corrData[corrData['delta_t']<=0.02]
        
        # get all the parameters from the file
        dataName='mean'+color
        stdName='std'+color
        
        # calculate fit to noise
        logdata=np.log10(corrData[stdName])
        logt=np.log10(corrData['delta_t'])
        pf=np.polyfit(logt,logdata,4)
        p=np.poly1d(pf)
        fitNoise=10**p(logt)

        data_list.append(corrData[dataName])
        std_list.append(fitNoise)
        b.add("C_"+filename,value=experiments['conc'+color+'(nM)'][i],vary=False)
        c.append("C_"+filename)
        
    data=np.array(data_list)
    std=np.array(std_list)
       
    # performing fits with 3-d Gaussian model
    bG=copy.deepcopy(b)
    bG.add('D',experiments['diffcoef'+color][0],vary=False)    
    bG.add('wxy',value=0.25,vary=True)
    bG.add('wz',value=0.7,vary=True)
    bG.add('slope',value=1.0,vary=False)

    # fit 3-d Gaussian
    resultG=minimize(g_all,bG,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultG.params,paraPickleFile)
    
    datasetG=makeResultDataFrame(resultG,{'color':color,'v':resultG.params['wxy'].value**2*resultG.params['wz'].value*m.pi**1.5,'chisq':resultG.chisqr,'redchi':resultG.redchi})
    datasetG.to_csv(datadir+color+'gaussian_SOME.csv')
    
    print fit_report(resultG)
    logfile.write(fit_report(resultG)+'\n')
    
    bG.add('F',0.05, min=0.0, max=0.5, vary=True)
    bG.add('tf',1e-6, min=1e-9, max=1e-4, vary=True)
    
    # fit 3-d Gaussian with triplet
    resultG=minimize(g_all_t,bG,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultG.params,paraPickleFile)

    datasetGt=makeResultDataFrame(resultG,{'color':color,'v':resultG.params['wxy'].value**2*resultG.params['wz'].value*m.pi**1.5,'chisq':resultG.chisqr,'redchi':resultG.redchi})
    datasetGt.to_csv(datadir+color+'gaussian_triplet_SOME.csv')
    
    print fit_report(resultG)
    logfile.write(fit_report(resultG)+'\n')

    # performing fits with numerical models
    dataNcolor=dataN[dataN['color']==color]
    dataSelect=pd.merge(dataNcolor,experiments,left_on='dataset',right_on='filename')
    
    print dataSelect
    
    bN=copy.deepcopy(b)
    bN.add('D',experiments['diffcoef'+color][0],vary=False)    
    bN.add('w0',dataSelect['w0'].mean(),vary=True)
    bN.add('a',experiments['fiber'+color+'(microns)'][0],vary=False)
    bN.add('r0',dataSelect['r0'].mean(),vary=True)
    bN.add('lambdaex',experiments['lambdaex'+color+'(nm)'][0]/1000.0,vary=False)
    bN.add('lambdaem',experiments['lambdaem'+color+'(nm)'][0]/1000.0,vary=False)
    bN.add('n',experiments['n'][0],vary=False)
    bN.add('slope',value=1.0,vary=False)
    
    # fit numerical model
    resultN=minimize(g_all_n,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultN.params,paraPickleFile)
    
    print fit_report(resultN)
    logfile.write(fit_report(resultN)+'\n')

    v1=vol1dict(resultN.params)
    v2=vol2dict(resultN.params)
    print "Volume = ",v1*v1/v2
        
    datasetGN=makeResultDataFrame(resultN,{'color':color,'v':v1*v1/v2,'chisq':resultN.chisqr,'redchi':resultN.redchi})
    datasetGN.to_csv(datadir+color+'numerical_SOME_rev.csv')

    # fit numerical model k_real
    resultN=minimize(g_all_nr,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultN.params,paraPickleFile)
    
    print fit_report(resultN)
    logfile.write(fit_report(resultN)+'\n')

    v1=vol1dict(resultN.params,cef=k_real)
    v2=vol2dict(resultN.params,cef=k_real)
    print "Volume = ",v1*v1/v2
        
    datasetGNR=makeResultDataFrame(resultN,{'color':color,'v':v1*v1/v2,'chisq':resultN.chisqr,'redchi':resultN.redchi})
    datasetGNR.to_csv(datadir+color+'numericalR_SOME_rev.csv')

    dataNTcolor=dataNT[dataNT['color']==color]
    dataSelect=pd.merge(dataNTcolor,experiments,left_on='dataset',right_on='filename')
    
    bN['w0'].value=dataSelect['w0'].mean()
    bN['r0'].value=dataSelect['r0'].mean()
    bN.add('F',dataSelect['F'].mean(), min=0.0, max=0.5, vary=True)
    bN.add('tf',dataSelect['tf'].mean(), min=1e-9, max=1e-4, vary=True)

    # fit numerical + triplet
    resultNR=minimize(g_all_nt,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultNR.params,paraPickleFile)
    
    print fit_report(resultNR)
    logfile.write(fit_report(resultNR)+'\n')

    v1=vol1dict(resultNR.params)
    v2=vol2dict(resultNR.params)
    print "Volume = ",v1*v1/v2
    
    datasetGNt=makeResultDataFrame(resultNR,{'color':color,'v':v1*v1/v2,'chisq':resultNR.chisqr,'redchi':resultNR.redchi})
    datasetGNt.to_csv(datadir+color+'numericalTriplet_SOME_rev.csv')
    
    # fit numerical + triplet
    resultNR=minimize(g_all_ntr,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(resultNR.params,paraPickleFile)
    
    print fit_report(resultNR)
    logfile.write(fit_report(resultNR)+'\n')

    v1=vol1dict(resultNR.params,cef=k_real)
    v2=vol2dict(resultNR.params,cef=k_real)
    print "Volume = ",v1*v1/v2
    
    datasetGNtR=makeResultDataFrame(resultNR,{'color':color,'v':v1*v1/v2,'chisq':resultNR.chisqr,'redchi':resultNR.redchi})
    datasetGNtR.to_csv(datadir+color+'numericalRTriplet_SOME_rev.csv')
    
    logfile.close()
    
paraPickleFile.close()
