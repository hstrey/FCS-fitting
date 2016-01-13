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

from GaussianModels import g_oligo_all,g_oligo_all_n
#from GaussianModels_old import g_oligo_all,g_oligo_all_n

#defines the location of the data
#datadir='../062415/50um/dilutions/SOME/'
#datafile='../061515/OL_N/Retest with New Gaussian Models/OL100ace'
#datadir2='../061515/OL_N/Retest with New Gaussian Models/'
datadir='../Dilutions/SOME/'
datafile='../062415/25um/OL_Na/OL100'
datadir2='../062415/25um/OL_Na/'

corrData=pd.read_csv(datafile+'.csv')
logfile=open(datadir2+'oligo_'+'Numerical_B3R6_mixa_test1.log',"w")

# load the parameters for each fit from the pickle file
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_B3R6.pkl','r') as paraPickleFile:
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
corrData=corrData[corrData['delta_t']<=10.0]

t=corrData['delta_t']
data=np.array([corrData['meanB'],corrData['meanR'],corrData['meanBR']])
dataStd=np.array([corrData['stdB'],corrData['stdR'],corrData['stdBR']])

b=Parameters()
b.add('D',value=70.0,vary=True)
b.add('C',value=2.0,vary=True)
b.add('F_b',value=bBlue['F'].value,vary=True)
b.add('tf_b',value=bBlue['tf'].value,vary=True)
b.add('F_r',value=bRed['F'].value,vary=True)
b.add('tf_r',value=bRed['tf'].value,vary=True)
b.add('delta_z',value=1.0,vary=True)

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
    b.add('lambdaex_b',value=bBlue['lambdaex'].value,vary=False)
    b.add('lambdaem_b',value=bBlue['lambdaem'].value,vary=False)
    b.add('lambdaex_r',value=bRed['lambdaex'].value,vary=False)
    b.add('lambdaem_r',value=bRed['lambdaem'].value,vary=False)
    b.add('n',value=bBlue['n'].value,vary=False)

    out=minimize(g_oligo_all_n,b,args=(t,data,dataStd))
    print fit_report(b)
    logfile.write(fit_report(b)+'\n')

    gfit_all=g_oligo_all_n(b,t)

logfile.close()

plt.figure()

plt.subplot(3,1,1)
plt.errorbar(t,data[0],yerr=dataStd[0],fmt='ob')
plt.plot(t,gfit_all[0],color='r')
plt.xscale('log')
plt.ylabel('Blue g(t)')
plt.title('Oligo Gaussian Fit B3R6')

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
plt.savefig(datadir2+'oligo_'+'Numerical_B3R6_mixa_test1.png', bbox_inches='tight')
plt.savefig(datadir2+'oligo_'+'Numerical_B3R6_mixa_test1.pdf', bbox_inches='tight')
    
