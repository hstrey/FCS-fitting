# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 00:10:52 2015

@author: hstrey
"""
import numpy as np
import math as m
from lmfit import Model
from scipy.integrate import quad, dblquad
from scipy.special import erfi

# 3-d Gaussian focus FCS model with triplet correction
def g_FCS(t,C,wxy,wz,D,F,tf):
    v=wxy*wxy*wz*m.pi**1.5
    N=C*6.022e-1*v
    triplet=(1-F+F*np.exp(-t/tf))/(1-F)
    return 1.0+triplet/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)

# 3-d Gaussian focus FCS model
def g(t,C,wxy,wz,D):
    v=wxy*wxy*wz*m.pi**1.5
    N=C*6.022e-1*v
    return 1.0+1.0/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)

# 3-d Gaussian focus FCS model with triplet correction
def g_t(t,C,wxy,wz,D,F,tf):
    v=wxy*wxy*wz*m.pi**1.5
    N=C*6.022e-1*v
    triplet=(1-F+F*np.exp(-t/tf))/(1-F)
    return 1.0+triplet/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)
    
modelFCS_t = Model(g_t,independent_vars=['t'])
modelFCS = Model(g,independent_vars=['t'])

# model for standard deviation
# empirically p=0.33
def noise(t,tc,a,b,p):
    x=t/tc
    return np.abs(a)/x+np.abs(b)/x**p
    
modelNoise=Model(noise,independent_vars=['t'])

#constants
maxz=200 # z integration range in microns

# determine hermite-gaussian integration intervals
xh,yh=np.polynomial.hermite.hermgauss(50)

def w2(z,w0,lambdaex,n):
    return w0*w0*(1+(lambdaex*z/m.pi/w0/w0/n)**2)

def k(z,a,r0,lambdaem,n):
    return 1-np.exp(-2*a*a/(r0*r0*(1+(lambdaem*z/m.pi/r0/r0/n)**2)))

#def gi(eta,xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return np.exp(-xi**2)*k(eta-np.sqrt(D*t)*xi,a,R0,lambdaem,n)*k(eta+np.sqrt(D*t)*xi,a,R0,lambdaem,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi,w0,lambdaex,n)+w2(eta+np.sqrt(D*t)*xi,w0,lambdaex,n))

def g_noexp(eta,xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
    return m.sqrt(m.pi)*k(eta-np.sqrt(D*t)*xi,a,R0,lambdaem,n)*k(eta+np.sqrt(D*t)*xi,a,R0,lambdaem,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi,w0,lambdaex,n)+w2(eta+np.sqrt(D*t)*xi,w0,lambdaex,n))

def gz1_hermite(xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
    return np.sum(g_noexp(xi,xh,t,D,w0,a,R0,lambdaex,lambdaem,n)*yh)

def g_hermite(t,D,w0,a,R0,lambdaex,lambdaem,n):
    return quad(gz1_hermite,0,200,args=(t,D,w0,a,R0,lambdaex,lambdaem,n))

#def gz1(xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gi,0,200+np.sqrt(D*t)*xi,args=(xi,t,D,w0,a,R0,lambdaex,lambdaem,n))[0]

#def g(t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gz1,0,200,args=(t,D,w0,a,R0,lambdaex,lambdaem,n))

def g0(z,w0,a,R0,lambdaex,lambdaem,n):
    return k(z,a,R0,lambdaem,n)**2/w2(z,w0,lambdaex,n)

# vol1 is integral over k(z) and the square normalizes the function g_hermite
# g(t)=g_hermite/vol1**2
def vol1(a,r0,lambdaem,n,C=None,D=None,lambdaex=None,w0=None,F=None,tf=None):
    return quad(k,0,maxz,args=(a,r0,lambdaem,n))[0]*m.pi

def vol2(w0,a,r0,lambdaex,lambdaem,n,C=None,D=None,F=None,tf=None):
    return m.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n))[0]

def g_n(t,D,C,w0,a,r0,lambdaex,lambdaem,n):
    v1=vol1(a,r0,lambdaem,n)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

    return np.array([1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)[0]/C/6.022e-1/v1/v1 for tt in t])

def g_nt(t,D,C,w0,a,r0,lambdaex,lambdaem,n,F,tf):
    v1=vol1(a,r0,lambdaem,n)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2, "F",F,"tf",tf
    
    return np.array([1+(1-F+F*np.exp(-tt/tf))/(1-F)*g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)[0]/C/6.022e-1/v1/v1 for tt in t])

modelFCS_n = Model(g_n,independent_vars=['t'])
modelFCS_nt = Model(g_nt,independent_vars=['t'])
