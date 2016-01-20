# -*- coding: utf-8 -*-
"""
Created on Sat 16, 2016

@author: hstrey

includes function and definitions that are common to all FCS fitting models
"""

import numpy as np
from lmfit import Model

#constants
maxz=10 # z integration range in microns

# determine hermite-gaussian integration intervals
hermite_order=32
xh,yh=np.polynomial.hermite.hermgauss(hermite_order)
xh_half=xh[hermite_order/2:]
yh_half=yh[hermite_order/2:]


# model for fitting standard deviation (noise)
def fitnoise(t,std):
    # calculate fit to data and fit to noise
    logdata=np.log10(std)
    logt=np.log10(t)
    pf=np.polyfit(logt,logdata,4)
    p=np.poly1d(pf)
    return 10**p(logt)

modelNoise=Model(noise,independent_vars=['t'])

# functions that define realistic MDF
def w2(z,w0,lambdaex,n):
    return w0*w0+(lambdaex*z/np.pi/w0/n)**2

def k(z,a,r0,lambdaem,n):
    return 1-np.exp(-2*a*a/(r0*r0+(lambdaem*z/np.pi/r0/n)**2))

def RR(z,r0,lambdaem,n):
    return r0*r0+(lambdaem*z/np.pi/r0/n)**2

def k_real(z,a,r0,lambdaem,n):
    rr=RR(z,r0,lambdaem,n)
    return np.where(rr>a*a,a*a/rr,1.0)


