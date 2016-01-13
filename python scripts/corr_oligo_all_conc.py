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

from GaussianModels import g_oligo_all,g_oligo_all_nc

#defines the location of the data
datadirdil='../sample data/dilution/'
datadir='../sample data/oligo/'
datafile='../sample data/oligo/O1250'

corrData=pd.read_csv(datafile+'.csv')

# load the parameters for each fit from the pickle file
parameters=collections.defaultdict(list)
with open(datadirdil+'corr_average_all.pkl','r') as paraPickleFile:
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['B'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))
     parameters['R'].append(pickle.load(paraPickleFile))

Gauss3D=False
GaussBeam=False
print "0 = 3D Gauss, 1 = 3D Gauss + triplet, 2 = Gaussian beam, 3 = Gaussian beam + triplet"
bluePick = raw_input("Pick calibration parameter set for Blue channel: ")
if bluePick=='0':
    Gauss3D=True
    bBlue=parameters['B'][0]
    bBlue.add('F',value=0.0,vary=False)
    bBlue.add('tf',value=1e-6,vary=False)
elif bluePick=='1':
    Gauss3D=True
    bBlue=parameters['B'][1]
elif bluePick=='2':
    GaussBeam=True
    bBlue=parameters['B'][2]
    bBlue.add('F',value=0.0,vary=False)
    bBlue.add('tf',value=1e-6,vary=False)
elif bluePick=='3':
     bBlue=parameters['B'][3]
   
redPick = raw_input("Pick calibration parameter set for Red channel: ")
if redPick=='0':
    Gauss3D=True
    bRed=parameters['R'][0]
    bRed.add('F',value=0.0,vary=False)
    bRed.add('tf',value=1e-6,vary=False)
elif redPick=='1':
    Gauss3D=True
    bRed=parameters['R'][1]
elif redPick=='2':
    GaussBeam=True
    bRed=parameters['R'][2]
    bRed.add('F',value=0.0,vary=False)
    bRed.add('tf',value=1e-6,vary=False)
elif bluePick=='3':
     bRed=parameters['R'][3]

#data set for fitting mean square displacements
corrData=corrData[corrData['delta_t']>=1e-6]
corrData=corrData[corrData['delta_t']<=0.1]

t=corrData['delta_t']
data=np.array([corrData['meanB'],corrData['meanR'],corrData['meanBR']])
dataStd=np.array([corrData['stdB'],corrData['stdR'],corrData['stdBR']])

logfile=open(datadir+'oligo_all_nc2.log',"w")

b=Parameters()
b.add('D',value=80,vary=True)
b.add('Cb',value=1.26,vary=True)
b.add('Cr',value=1.25,vary=True)
b.add('Cc',expr='Cr if Cr<Cb else Cb')
b.add('F_b',value=bBlue['F'].value,vary=False)
b.add('tf_b',value=bBlue['tf'].value,vary=False)
b.add('F_r',value=bRed['F'].value,vary=False)
b.add('tf_r',value=bRed['tf'].value,vary=False)
b.add('delta_z',value=0.5,vary=True)

if Gauss3D:
    #combine parameters from blue and red into one
    b.add('wz_b',value=bBlue['wz'].value,vary=False)
    b.add('wz_r',value=bRed['wz'].value,vary=False)
    b.add('wxy_b',value=bBlue['wxy'].value,vary=False)
    b.add('wxy_r',value=bRed['wxy'].value,vary=False)
   
    out=minimize(g_oligo_all,b,args=(t,data,dataStd))
    print fit_report(b)
    logfile.write(fit_report(b)+'\n')
    
    gfit_all=g_oligo_all(b,t)
else:
    #combine parameters from blue and red into one
    b.add('w0_b',value=bBlue['w0'].value,vary=False)
    b.add('w0_r',value=bRed['w0'].value,vary=False)
    b.add('a_b',value=bBlue['a'].value,vary=False)
    b.add('a_r',value=bRed['a'].value,vary=False)
    b.add('r0_b',value=bBlue['r0'].value,vary=False)
    b.add('r0_r',value=bRed['r0'].value,vary=False)
    b.add('D', 69.2, True, None, None, None)
    b.add('lambdaex_b',value=bBlue['lambdaex'].value,vary=False)
    b.add('lambdaem_b',value=bBlue['lambdaem'].value,vary=False)
    b.add('lambdaex_r',value=bRed['lambdaex'].value,vary=False)
    b.add('lambdaem_r',value=bRed['lambdaem'].value,vary=False)
    b.add('n',value=bBlue['n'].value,vary=False)
    
    out=minimize(g_oligo_all_nc,b,args=(t,data,dataStd))
    print fit_report(b)
    logfile.write(fit_report(b)+'\n')   
    gfit_all=g_oligo_all_nc(b,t)

logfile.close()

plt.figure()

plt.subplot(3,1,1)
plt.errorbar(t,data[0],yerr=dataStd[0],fmt='ob')
plt.plot(t,gfit_all[0])
plt.xscale('log')
plt.ylabel('Blue g(t)')

plt.subplot(3,1,2)
plt.errorbar(t,data[1],yerr=dataStd[1],fmt='or')
plt.plot(t,gfit_all[1])
plt.xscale('log')
plt.ylabel('Red g(t)')

plt.subplot(3,1,3)
plt.errorbar(t,data[2],yerr=dataStd[2],fmt='co')
plt.plot(t,gfit_all[2])
plt.xscale('log')
plt.ylabel('Blue-Red g(t)')
       
