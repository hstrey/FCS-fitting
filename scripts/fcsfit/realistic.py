import numpy as np
from lmfit import Model
from scipy.integrate import quad

from common import w2,k,RR,k_real,xh_half,yh_half

#####################################################################
# Numerical models
#####################################################################

def g_noexp(eta,xi,t,D,w0,a,R0,lambdaex,lambdaem,n,k):
    return np.sqrt(np.pi)*k(eta-np.sqrt(D*t)*xi,a,R0,lambdaem,n)*k(eta+np.sqrt(D*t)*xi,a,R0,lambdaem,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi,w0,lambdaex,n)+w2(eta+np.sqrt(D*t)*xi,w0,lambdaex,n))

def gz1_hermite(xi,t,D,w0,a,R0,lambdaex,lambdaem,n,k):
    return 2*np.sum(g_noexp(xi,xh_half,t,D,w0,a,R0,lambdaex,lambdaem,n,k)*yh_half)

def g_hermite(t,D,w0,a,R0,lambdaex,lambdaem,n,k=k):
    return quad(gz1_hermite,0,maxz,args=(t,D,w0,a,R0,lambdaex,lambdaem,n,k))

def g0(z,w0,a,R0,lambdaex,lambdaem,n,k=k):
    return k(z,a,R0,lambdaem,n)**2/w2(z,w0,lambdaex,n)

# vol1 is integral over k(z) and the square normalizes the function g_hermite
# g(t)=g_hermite/vol1**2
def vol1(a,r0,lambdaem,n,C=None,D=None,lambdaex=None,w0=None,F=None,tf=None,k=k):
    return quad(k,0,maxz,args=(a,r0,lambdaem,n))[0]*np.pi

def vol2(w0,a,r0,lambdaex,lambdaem,n,C=None,D=None,F=None,tf=None,k=k):
    return np.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n,k))[0]

def vol1dict(b,k=k):
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value

    return quad(k,0,maxz,args=(a,r0,lambdaem,n))[0]*np.pi

def vol2dict(b,k=k):
    w0=b['w0'].value
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value
    lambdaex=b['lambdaex'].value
    return np.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n,k))[0]

def g_n(t,D,C,w0,a,r0,lambdaex,lambdaem,n,k=k):
    v1=vol1(a,r0,lambdaem,n,k)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

    return np.array([1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,k)[0]/C/6.022e-1/v1/v1 for tt in t])

def g_n_norm(t,D,w0,a,r0,lambdaex,lambdaem,n,k=k):
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)
    return np.array([1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,k)[0]/v2 for tt in t])

def g_nt(t,D,C,w0,a,r0,lambdaex,lambdaem,n,F,tf,k=k):
    v1=vol1(a,r0,lambdaem,n,k)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2, "F",F,"tf",tf

    return np.array([1+(1-F+F*np.exp(-tt/tf))/(1-F)*g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,k)[0]/C/6.022e-1/v1/v1 for tt in t])

modelFCS_n = Model(g_n,independent_vars=['t'])
modelFCS_nt = Model(g_nt,independent_vars=['t'])
modelFCS_nr = Model(g_n,independent_vars=['t'],k=k_real)
modelFCS_ntr = Model(g_nt,independent_vars=['t'],k=k_real)
