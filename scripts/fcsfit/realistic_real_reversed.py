import numpy as np
from lmfit import Model
from scipy.integrate import quad

from common import w2,RR,maxz,xh_half,yh_half

#####################################################################
# Numerical models
#####################################################################
def k_real(z,a,r0,lambdaem,n):
    rr=RR(z,r0,lambdaem,n)
    return a*a/rr

def g_noexp(eta,xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
    return np.sqrt(np.pi)*k_real(eta-np.sqrt(D*t)*xi,a,R0,lambdaem,n)*k_real(eta+np.sqrt(D*t)*xi,a,R0,lambdaem,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi,w0,lambdaex,n)+w2(eta+np.sqrt(D*t)*xi,w0,lambdaex,n))

def g_hermite(t,D,w0,a,R0,lambdaex,lambdaem,n):
    return 2*np.sum([gz1(x,t,D,w0,a,R0,lambdaex,lambdaem,n) for x in xh_half]*yh_half)

def gz1(xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
    return quad(g_noexp,0,maxz,args=(xi,t,D,w0,a,R0,lambdaex,lambdaem,n))[0]

#def gz1(xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gi,0,200+np.sqrt(D*t)*xi,args=(xi,t,D,w0,a,R0,lambdaex,lambdaem,n))[0]

#def g(t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gz1,0,200,args=(t,D,w0,a,R0,lambdaex,lambdaem,n))

def g0(z,w0,a,R0,lambdaex,lambdaem,n):
    return k_real(z,a,R0,lambdaem,n)**2/w2(z,w0,lambdaex,n)

def g0_1(z,w0,a,R0,lambdaex,lambdaem,n):
    return 1.0/w2(z,w0,lambdaex,n)

# vol1 is integral over k(z) and the square normalizes the function g_hermite
# g(t)=g_hermite/vol1**2
# !!!! change code to allow for k_real !!!
def vol1r(a,r0,lambdaem,n,C=None,D=None,lambdaex=None,w0=None,F=None,tf=None):
    # have to find the z for which rr=a*a
    z_1=np.sqrt(a*a-r0*r0)/lambdaem*np.pi*r0*n
    return (z_1+quad(k_real,z_1,maxz,args=(a,r0,lambdaem,n))[0])*np.pi

def vol2r(w0,a,r0,lambdaex,lambdaem,n,C=None,D=None,F=None,tf=None):
    # have to find the z for which rr=a*a
    z_1=np.sqrt(a*a-r0*r0)/lambdaem*np.pi*r0*n
    return np.pi/2.0*(quad(g0_1,0,z_1,args=(w0,a,r0,lambdaex,lambdaem,n))[0]+quad(g0,z_1,maxz,args=(w0,a,r0,lambdaex,lambdaem,n))[0])

def vol1dict(b):
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value

    return quad(k_real,0,maxz,args=(a,r0,lambdaem,n))[0]*np.pi

def vol2dict(b):
    w0=b['w0'].value
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value
    lambdaex=b['lambdaex'].value
    return np.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n))[0]

def g_nr(t,D,C,w0,a,r0,lambdaex,lambdaem,n):
    v1=vol1(a,r0,lambdaem,n,k)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

    return np.array([1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)/C/6.022e-1/v1/v1 for tt in t])

def g_nr_norm(t,D,w0,a,r0,lambdaex,lambdaem,n):
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)
    return np.array([1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)/v2 for tt in t])

def g_ntr(t,D,C,w0,a,r0,lambdaex,lambdaem,n,F,tf):
    v1=vol1(a,r0,lambdaem,n,k)
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,k)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2, "F",F,"tf",tf

    return np.array([1+(1-F+F*np.exp(-tt/tf))/(1-F)*g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)/C/6.022e-1/v1/v1 for tt in t])

modelFCS_nr = Model(g_nr,independent_vars=['t'])
modelFCS_ntr = Model(g_ntr,independent_vars=['t'])
