# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:32:43 2015

@author: hstrey
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import pandas as pd
import sys,os

# make sure that this scripts can find the fcsfit folder that is one level below
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from fcsfit.common import k_real
from fcsfit.realistic_reversed import vol1, vol2
from fcsfit.realistic_rev_oligo import vol2c, vol2f

#defines the location of the data
datadir='../../data/dilutions/SOME/'
datadir_all='../../data/dilutions/'

# load the parameters for each fit from the pickle file
# there are 6 parameter objects per color 3dG, 3dGt, n, nt, nr, ntr
parameters=collections.defaultdict(list)
with open(datadir+'corr_average_all_final2.pkl',"r") as paraPickleFile:
    for i in range(6):
        parameters['B'].append(pickle.load(paraPickleFile))
    for i in range(6):
        parameters['R'].append(pickle.load(paraPickleFile))

datadict={}

bBlue=parameters['B'][2]
bRed=parameters['R'][2]

w1=bBlue['w0'].value
w2=bRed['w0'].value
r1=bBlue['r0'].value
r2=bRed['r0'].value
a1=value=bBlue['a'].value
a2=value=bRed['a'].value
lambdaex1=0.488
lambdaem1=0.519
lambdaex2=0.633
lambdaem2=0.657
n=1.33

print "w1 = ",w1
print "w2 = ",w2
print "r1 = ",r1
print "r2 = ",r2
print "a1 = ",a1
print "a2 = ",a2

v1b=vol1(a1,r1,lambdaem1,n)
v2b=vol2(w1,a1,r1,lambdaex1,lambdaem1,n)
print "Volume Blue = ",v1b*v1b/v2b

v1r=vol1(a2,r2,lambdaem2,n)
v2r=vol2(w2,a2,r2,lambdaex2,lambdaem2,n)
print "Volume red = ",v1r*v1r/v2r

dz_arrayN=np.linspace(-2,2,51)
datadict['dz']=dz_arrayN

v2c=np.array([vol2c(w1,w2,a1,a2,r1,r2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz) for dz in dz_arrayN ])
v2f=np.array([vol2f(w1,w2,a1,a2,r1,r2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz) for dz in dz_arrayN ])
vol_c=v1b*v1r/v2c
vol_f=v1b*v1r/v2f
datadict['vol_c']=vol_c
datadict['vol_f']=vol_f

bBlue=parameters['B'][4]
bRed=parameters['R'][4]

w1=bBlue['w0'].value
w2=bRed['w0'].value
r1=bBlue['r0'].value
r2=bRed['r0'].value
a1=value=bBlue['a'].value
a2=value=bRed['a'].value
lambdaex1=0.488
lambdaem1=0.519
lambdaex2=0.633
lambdaem2=0.657
n=1.33

print "w1 = ",w1
print "w2 = ",w2
print "r1 = ",r1
print "r2 = ",r2
print "a1 = ",a1
print "a2 = ",a2

v1b=vol1(a1,r1,lambdaem1,n)
v2b=vol2(w1,a1,r1,lambdaex1,lambdaem1,n)
print "Volume Blue = ",v1b*v1b/v2b

v1r=vol1(a2,r2,lambdaem2,n)
v2r=vol2(w2,a2,r2,lambdaex2,lambdaem2,n)
print "Volume red = ",v1r*v1r/v2r

v2cr=np.array([vol2c(w1,w2,a1,a2,r1,r2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k=k_real) for dz in dz_arrayN ])
v2fr=np.array([vol2f(w1,w2,a1,a2,r1,r2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k=k_real) for dz in dz_arrayN ])
vol_cr=v1b*v1r/v2cr
vol_fr=v1b*v1r/v2fr
datadict['vol_cr']=vol_cr
datadict['vol_fr']=vol_fr

plt.figure()
plt.plot(dz_arrayN,vol_c)
plt.plot(dz_arrayN,vol_f)
plt.xlabel('delta z in micrometer')
plt.ylabel('Veff in micrometer^3')
plt.savefig(datadir+'NumericalVeff.png', bbox_inches='tight')
plt.savefig(datadir+'NumericalVeff.pdf', bbox_inches='tight')

plt.figure()
plt.plot(dz_arrayN,vol_cr)
plt.plot(dz_arrayN,vol_fr)
plt.xlabel('delta z in micrometer')
plt.ylabel('VeffR in micrometer^3')
plt.savefig(datadir+'NumericalVeffR.png', bbox_inches='tight')
plt.savefig(datadir+'NumericalVeffR.pdf', bbox_inches='tight')

bBlue=parameters['B'][0]
bRed=parameters['R'][0]

w1=bBlue['wxy'].value
w2=bRed['wxy'].value
z1=bBlue['wz'].value
z2=bRed['wz'].value
lambdaex1=0.488
lambdaem1=0.519
lambdaex2=0.633
lambdaem2=0.657
n=1.33

print "w1 = ",w1
print "w2 = ",w2
print "z1 = ",z1
print "z2 = ",z2

print "Volume Blue = ",np.pi**1.5*w1**2*z1
print "Volume red = ",np.pi**1.5*w2**2*z2

dz_arrayG=np.linspace(-4,4,51)
datadict['dz_G']=dz_arrayG

volG_c=np.pi**1.5*(w1**2+w2**2)*np.sqrt(z1**2+z2**2)/2**1.5*np.exp(dz_arrayG**2/(z1**2+z2**2))
datadict['volG_c']=volG_c

plt.figure()
plt.plot(dz_arrayG,volG_c)
plt.xlabel('delta z in micrometer')
plt.ylabel('Veff in micrometer^3')
plt.savefig(datadir+'GaussianVeff.png', bbox_inches='tight')
plt.savefig(datadir+'GaussianVeff.pdf', bbox_inches='tight')

dataOut=pd.DataFrame(datadict)
dataOut.to_csv(datadir_all+'Veff_dz.csv')
