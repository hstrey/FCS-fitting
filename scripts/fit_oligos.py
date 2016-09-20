# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:57:09 2015

@author: hstrey
"""

import numpy as np
from lmfit import Parameters, minimize, fit_report
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import collections

from fcsfit.common import k_real, fitnoise
from fcsfit.gaussian_oligo import g_oligo_all
from fcsfit.realistic_rev_oligo import g_oligo_all_n

#defines the location of the data
datadir='../data/dilutions/SOME/'
datadir_all='../data/dilutions/RAW/'
oligodir='../data/oligos/'
datafile='OL100a'

datadict={}

corrData=pd.read_csv(oligodir+datafile+'.csv')

# load the parameters for each fit from the pickle file
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_all_final2.pkl',"r") as paraPickleFile:
    for i in range(6):
        parameters['B'].append(pickle.load(paraPickleFile))
    for i in range(6):
        parameters['R'].append(pickle.load(paraPickleFile))

Gauss3D=False
GaussBeam=False
print """0 = 3D Gauss
1 = 3D Gauss + triplet
2 = Gaussian beam
3 = Gaussian beam k_real
4 = Gaussian beam + triplet
5 = Gaussian beam k_real + triplet"""

bluePick = int(raw_input("Pick calibration parameter set for Blue channel: "))
bBlue=parameters['B'][bluePick]

redPick = int(raw_input("Pick calibration parameter set for Red channel: "))
bRed=parameters['R'][redPick]

logfile=open(datadir_all+'oligo_'+str(bluePick)+'_'+str(redPick)+'.log',"w")

#data set for fitting mean square displacements
corrData=corrData[corrData['delta_t']>=1e-5]
corrData=corrData[corrData['delta_t']<=0.1]

t=corrData['delta_t']
data=np.array([corrData['meanB'],corrData['meanR'],corrData['meanBR']])
stdB_fit=fitnoise(t,corrData['stdB'])
stdR_fit=fitnoise(t,corrData['stdR'])
stdBR_fit=fitnoise(t,corrData['stdBR'])
dataStd=np.array([corrData['stdB'],corrData['stdR'],corrData['stdBR']])
dataStd_fit=np.array([stdB_fit,stdR_fit,stdBR_fit])

# save data, stderr and fitnoise just in case
datadict['t']=t
datadict['oligoB']=np.array(corrData['meanB'])
datadict['oligoR']=np.array(corrData['meanR'])
datadict['oligoBR']=np.array(corrData['meanBR'])
datadict['oligoB_std']=np.array(corrData['stdB'])
datadict['oligoR_std']=np.array(corrData['stdR'])
datadict['oligoBR_std']=np.array(corrData['stdBR'])
datadict['oligoB_stdfit']=stdB_fit
datadict['oligoR_stdfit']=stdR_fit
datadict['oligoBR_stdfit']=stdBR_fit

b=Parameters()
b.add('D',value=70.0,vary=True)
b.add('C',value=2.3,vary=True)
b.add('delta_z',value=3.9,vary=True)

# if triplet
if bluePick==1 or bluePick==4 or bluePick==5:
    b.add('F_b',value=bBlue['F'].value,min=0.0,max=0.5,vary=False)
    b.add('tf_b',value=bBlue['tf'].value,min=1e-9,max=1e-4,vary=False)
else:
    b.add('F_b',0.0,vary=False)
    b.add('tf_b',1e-6,vary=False)

if redPick==1 or redPick==4 or redPick==5:
    b.add('F_r',value=bRed['F'].value,min=0.0,max=0.5,vary=False)
    b.add('tf_r',value=bRed['tf'].value,min=1e-9,max=1e-4,vary=False)
else:
    b.add('F_r',0.0,vary=False)
    b.add('tf_r',1e-6,vary=False)

# if Gauss3D
if (bluePick==1 or bluePick==0) and (redPick==1 or redPick==0):
    #combine parameters from blue and red into one
    b.add('wz_b',value=bBlue['wz'].value,vary=False)
    b.add('wz_r',value=bRed['wz'].value,vary=False)
    b.add('wxy_b',value=bBlue['wxy'].value,vary=False)
    b.add('wxy_r',value=bRed['wxy'].value,vary=False)

    out=minimize(g_oligo_all,b,args=(t,data,dataStd_fit))
    print fit_report(out)
    logfile.write(fit_report(out)+'\n')
    gfit_all=g_oligo_all(out.params,t)
    datadict['fitB']=gfit_all[0]
    datadict['fitR']=gfit_all[1]
    datadict['fitBR']=gfit_all[2]

elif (5>=bluePick>=2) and (5>=redPick>=2):
    #combine parameters from blue and red into one
    b.add('w0_b',value=bBlue['w0'].value,vary=False)
    b.add('w0_r',value=bRed['w0'].value,vary=False)
    b.add('a_b',value=bBlue['a'].value,vary=False)
    b.add('a_r',value=bRed['a'].value,vary=False)
    b.add('r0_b',value=bBlue['r0'].value,vary=False)
    b.add('r0_r',value=bRed['r0'].value,vary=False)
    b.add('lambdaex_b',value=bBlue['lambdaex'].value,vary=False)
    b.add('lambdaem_b',value=bBlue['lambdaem'].value,vary=False)
    b.add('lambdaex_r',value=bRed['lambdaex'].value,vary=False)
    b.add('lambdaem_r',value=bRed['lambdaem'].value,vary=False)
    b.add('n',value=bBlue['n'].value,vary=False)
    
    if (bluePick==3 or bluePick==5):
        out=minimize(g_oligo_all_n,b,args=(t,data,dataStd_fit,k_real))
    else:
        out=minimize(g_oligo_all_n,b,args=(t,data,dataStd_fit))
        
    print fit_report(out)
    logfile.write(fit_report(out)+'\n')

    gfit_all=g_oligo_all_n(out.params,t)
    datadict['fitB']=gfit_all[0]
    datadict['fitR']=gfit_all[1]
    datadict['fitBR']=gfit_all[2]

logfile.close()
dataFits=pd.DataFrame(datadict)
dataFits.to_csv(oligodir+'oligo_fits_'+str(bluePick)+'_'+str(redPick)+'.csv')

plt.figure()
plt.title('Oligo Fit')

plt.subplot(3,1,1)
plt.errorbar(t,data[0],yerr=dataStd[0],fmt='ob')
plt.plot(t,gfit_all[0],color='r')
plt.xscale('log')
plt.ylabel('Blue g(t)')

plt.subplot(3,1,2)
plt.errorbar(t,data[1],yerr=dataStd[1],fmt='or')
plt.plot(t,gfit_all[1],color='b')
plt.xscale('log')
plt.ylabel('Red g(t)')

plt.subplot(3,1,3)
plt.errorbar(t,data[2],yerr=dataStd[2],fmt='co')
plt.plot(t,gfit_all[2])
plt.xscale('log')
plt.ylabel('Blue-Red g(t)')
plt.savefig(oligodir+'oligo_'+str(bluePick)+'_'+str(redPick)+'.png', bbox_inches='tight')
plt.savefig(oligodir+'oligo_'+str(bluePick)+'_'+str(redPick)+'.pdf', bbox_inches='tight')
plt.show()

