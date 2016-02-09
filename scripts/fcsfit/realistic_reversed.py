import numpy as np
from lmfit import Model
from scipy.integrate import quad

from common import w2,k,k_real,maxz,xh_half,yh_half

#####################################################################
# Numerical models
#####################################################################


def g_noexp(eta,xi,t,D,w0,a,R0,lambdaex,lambdaem,n,mdf):
    return np.sqrt(np.pi)*mdf(eta-np.sqrt(D*t)*xi,a,R0,lambdaem,n)*mdf(eta+np.sqrt(D*t)*xi,a,R0,lambdaem,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi,w0,lambdaex,n)+w2(eta+np.sqrt(D*t)*xi,w0,lambdaex,n))

def g_hermite(t,D,w0,a,R0,lambdaex,lambdaem,n,mdf=k):
    return 2*np.sum([gz1(x,t,D,w0,a,R0,lambdaex,lambdaem,n,mdf) for x in xh_half]*yh_half)

def gz1(xi,t,D,w0,a,R0,lambdaex,lambdaem,n,mdf):
    return quad(g_noexp,0,maxz,args=(xi,t,D,w0,a,R0,lambdaex,lambdaem,n,mdf))[0]

#def gz1(xi,t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gi,0,200+np.sqrt(D*t)*xi,args=(xi,t,D,w0,a,R0,lambdaex,lambdaem,n))[0]

#def g(t,D,w0,a,R0,lambdaex,lambdaem,n):
#    return quad(gz1,0,200,args=(t,D,w0,a,R0,lambdaex,lambdaem,n))

def g0(z,w0,a,R0,lambdaex,lambdaem,n,mdf):
    return mdf(z,a,R0,lambdaem,n)**2/w2(z,w0,lambdaex,n)

# vol1 is integral over k(z) and the square normalizes the function g_hermite
# g(t)=g_hermite/vol1**2
# !!!! change code to allow for k_real !!!
def vol1(a,r0,lambdaem,n,C=None,D=None,lambdaex=None,w0=None,F=None,tf=None,mdf=k):
    return quad(mdf,0,maxz,args=(a,r0,lambdaem,n))[0]*np.pi

def vol2(w0,a,r0,lambdaex,lambdaem,n,C=None,D=None,F=None,tf=None,mdf=k):
    return np.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n,mdf))[0]

def vol1dict(b,mdf=k):
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value

    return quad(mdf,0,maxz,args=(a,r0,lambdaem,n))[0]*np.pi

def vol2dict(b,k=k):
    w0=b['w0'].value
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value
    lambdaex=b['lambdaex'].value
    return np.pi/2.0*quad(g0,0,maxz,args=(w0,a,r0,lambdaex,lambdaem,n,k))[0]

def g_n(t,D,C,w0,a,r0,lambdaex,lambdaem,n,mdf=k):
    v1=vol1(a,r0,lambdaem,n,mdf=mdf)**2
    v12=v1*v1
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,mdf=mdf)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

    return 1.0+np.array([g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,mdf) for tt in t])/C/6.022e-1/v12

def g_n_norm(t,D,w0,a,r0,lambdaex,lambdaem,n,mdf=k):
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,mdf=mdf)
    return np.array([g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,mdf) for tt in t])/v2

def g_nt(t,D,C,w0,a,r0,lambdaex,lambdaem,n,F,tf,mdf=k):
    v1=vol1(a,r0,lambdaem,n,mdf=mdf)
    v12=v1*v1
    v2=vol2(w0,a,r0,lambdaex,lambdaem,n,mdf=mdf)

    print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2, "F",F,"tf",tf

    return 1.0+np.array([(1-F+F*np.exp(-tt/tf))/(1-F)*g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n,mdf) for tt in t])/C/6.022e-1/v12

modelFCS_n = Model(g_n,independent_vars=['t'])
modelFCS_nt = Model(g_nt,independent_vars=['t'])
modelFCS_nr = Model(g_n,independent_vars=['t'],mdf=k_real)
modelFCS_ntr = Model(g_nt,independent_vars=['t'],mdf=k_real)
