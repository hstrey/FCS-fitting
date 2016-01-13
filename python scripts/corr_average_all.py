import numpy as np
from lmfit import Parameters, minimize, fit_report
import pandas as pd
import math as m
import pickle

from GaussianModels import g_all,g_all_t, g_all_n,g_all_nt,vol1dict, vol2dict

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
datadir='../sample data/dilution/'

data3dG=pd.read_csv(datadir+'gaussian.csv')
data3dGT=pd.read_csv(datadir+'gaussian_triplet.csv')
dataN=pd.read_csv(datadir+'Numerical.csv')
dataNT=pd.read_csv(datadir+'NumericalTriplet.csv')

paraPickleFile=open(datadir+'corr_average_all.pkl',"w")

for color in ['B','R']:
    data_list=[]
    std_list=[]
    c=[]
    b=Parameters()
    experiments=pd.read_table(datadir+'Some'+color+'.txt')
    logfile=open(datadir+color+'_all.log',"w")

    for i in range(len(experiments)):
        filename=experiments['filename'][i]
        corrSet=pd.read_csv(datadir+filename+'.csv')
        
        #data set for fitting mean square displacements
        corrData=corrSet[corrSet['delta_t']>=6e-8]
        corrData=corrData[corrData['delta_t']<=0.05]
        
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
    
    b.add('D',experiments['diffcoef'+color][0],vary=False)
    
    # performing fits with 3-d Gaussian model
    bG=b.copy()
    bG.add('wxy',value=0.25,vary=True)
    bG.add('wz',value=0.4,vary=True)

    minimize(g_all,bG,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(bG,paraPickleFile)
    
    datasetG=makeResultDataFrame(bG,{'color':color,'v':bG['wxy'].value**2*bG['wz'].value*m.pi**1.5})
    datasetG.to_csv(datadir+color+'gaussian_all.csv')
    
    print fit_report(bG)
    logfile.write(fit_report(bG)+'\n')
    
    bG.add('F',0.1,vary=True)
    bG.add('tf',1e-6,vary=True)
    
    minimize(g_all_t,bG,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(bG,paraPickleFile)

    datasetGt=makeResultDataFrame(bG,{'color':color,'v':bG['wxy'].value**2*bG['wz'].value*m.pi**1.5})
    datasetGt.to_csv(datadir+color+'gaussian_triplet_all.csv')
    
    print fit_report(bG)
    logfile.write(fit_report(bG)+'\n')

    # performing fits with numerical models
    dataNcolor=dataN[dataN['color']==color]
    dataSelect=pd.merge(dataNcolor,experiments,left_on='dataset',right_on='filename')
    
    print dataSelect
    
    bN=b.copy()
    bN.add('w0',dataSelect['w0'].mean(),vary=True)
    bN.add('a',experiments['fiber'+color+'(microns)'][0],vary=False)
    bN.add('r0',dataSelect['r0'].mean(),vary=True)
    bN.add('lambdaex',experiments['lambdaex'+color+'(nm)'][0]/1000.0,vary=False)
    bN.add('lambdaem',experiments['lambdaem'+color+'(nm)'][0]/1000.0,vary=False)
    bN.add('n',experiments['n'][0],vary=False)
    
    minimize(g_all_n,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(bN,paraPickleFile)
    
    print fit_report(bN)
    logfile.write(fit_report(bN)+'\n')

    v1=vol1dict(bN)
    v2=vol2dict(bN)
    print "Volume = ",v1*v1/v2
        
    datasetGN=makeResultDataFrame(bN,{'color':color,'v':v1*v1/v2})
    datasetGN.to_csv(datadir+color+'numerical_all.csv')

    dataNTcolor=dataNT[dataNT['color']==color]
    dataSelect=pd.merge(dataNTcolor,experiments,left_on='dataset',right_on='filename')
    
    bN['w0'].value=dataSelect['w0'].mean()
    bN['r0'].value=dataSelect['r0'].mean()
    bN.add('F',dataSelect['F'].mean(),vary=True)
    bN.add('tf',dataSelect['tf'].mean(),vary=True)
    minimize(g_all_nt,bN,args=(corrData['delta_t'],c,data,std))
    
    pickle.dump(bN,paraPickleFile)
    
    print fit_report(bN)
    logfile.write(fit_report(bN)+'\n')

    v1=vol1dict(bN)
    v2=vol2dict(bN)
    print "Volume = ",v1*v1/v2
    
    datasetGNt=makeResultDataFrame(bN,{'color':color,'v':v1*v1/v2})
    datasetGNt.to_csv(datadir+color+'numericalTriplet_all.csv')
    logfile.close()
    
paraPickleFile.close()
