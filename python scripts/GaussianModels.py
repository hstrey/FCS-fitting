# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 00:10:52 2015

@author: hstrey

lmfit Models for fitting FCS correlation function

modelFCS : 3-d Gaussian FCS model with parameters wx, wz, D diffusion coefficient, C concentration (all units are micrometer, seconds)

modelFCS_t: same as modelFCS with additional triplet contribution.  Additional parameters F triplet fraction, tf relaxation time

modelFCS_n: FCS fit function for more a realistic confocal detection function (Gaussian in x,y but not in z).  Has to be evaluated numerically.
Parameters: a pinhole radius, lambdaex, lambdaem, n index of refraction of immersion liquid, w0 width in x-y, D, C

modelFCS_nt: same as modelFCS but with triplet contribution

"""
import numpy as np
import math as m
from lmfit import Model
from scipy.integrate import quad

#####################################################################
# 3-d Gaussian models
#####################################################################

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

#####################################################################
# Numerical models
#####################################################################
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

def vol1dict(b):
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value

    return quad(k,0,maxz,args=(a,r0,lambdaem,n))[0]*m.pi

def vol2dict(b):
    w0=b['w0'].value   
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaem=b['lambdaem'].value   
    lambdaex=b['lambdaex'].value   
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

#####################################################################
# Combined fit dilutions residual functions
#####################################################################

def g_all(b,t,c,data=None,sigma=None):
    #C=b['C'].value
    corr_g=None
    D=b['D'].value
    wxy=b['wxy'].value
    wz=b['wz'].value
    for i,conc in enumerate(c):
        C=b[conc].value
        v=wxy*wxy*wz*m.pi**1.5
        N=C*6.022e-1*v
        g=1+1/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))

    print "wxy = ", wxy
    print "wz = ", wz
    print "SP = ", wz/wxy

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
    
def g_all_t(b,t,c,data=None,sigma=None):
    #C=b['C'].value
    corr_g=None
    D=b['D'].value
    wxy=b['wxy'].value
    wz=b['wz'].value
    F=b['F'].value
    tf=b['tf'].value
    for i,conc in enumerate(c):
        C=b[conc].value
        v=wxy*wxy*wz*m.pi**1.5
        N=C*6.022e-1*v
        g=1+(1-F+F*np.exp(-t/tf))/(1-F)/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))

    print "wxy = ", wxy
    print "wz = ", wz
    print "SP = ", wz/wxy

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
    
def g_all_n(b,t,c,data=None,sigma=None):
    corr_g=None
    D=b['D'].value
    w0=b['w0'].value
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaex=b['lambdaex'].value
    lambdaem=b['lambdaem'].value

    for i,conc in enumerate(c):
        C=6.022e-1*b[conc].value

        v1=vol1(a,r0,lambdaem,n)
        v2=vol2(w0,a,r0,lambdaex,lambdaem,n)

        print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

        g=[1+g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)[0]/C/v1/v1 for tt in t]
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))
    
    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
    
def g_all_nt(b,t,c,data=None,sigma=None):
    corr_g=None
    D=b['D'].value
    w0=b['w0'].value
    n = b['n'].value
    a=b['a'].value
    r0=b['r0'].value
    lambdaex=b['lambdaex'].value
    lambdaem=b['lambdaem'].value
    F=b['F'].value
    tf=b['tf'].value

    for i,conc in enumerate(c):
        C=6.022e-1*b[conc].value

        v1=vol1(a,r0,lambdaem,n)
        v2=vol2(w0,a,r0,lambdaex,lambdaem,n)

        print "w0 = ",w0,"R0 = ",r0,"c = ",C,"vol",v1*v1/v2

        g=[1+(1-F+F*np.exp(-tt/tf))/(1-F)*g_hermite(tt,D,w0,a,r0,lambdaex,lambdaem,n)[0]/C/v1/v1 for tt in t]
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))
    
    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
    
#####################################################################
# Combined fit oligo functions
#####################################################################

def g_oligo_all(b,t,data=None,sigma=None):
    D=b['D'].value
    wxy_b=b['wxy_b'].value
    wz_b=b['wz_b'].value
    wxy_r=b['wxy_r'].value
    wz_r=b['wz_r'].value
    delta_z=b['delta_z'].value
    C=b['C'].value
    F_b=b['F_b'].value
    F_r=b['F_r'].value
    tf_b=b['tf_b'].value
    tf_r=b['tf_r'].value
    
    vb=wxy_b*wxy_b*wz_b*m.pi**1.5
    N=6.022e-1*C*vb
    g=1.0+(1-F_b+F_b*np.exp(-t/tf_b))/(1-F_b)/N/(1+4*D*t/wxy_b/wxy_b)/np.sqrt(1+4*D*t/wz_b/wz_b)
    corr_g=g[:]
    
    vr=wxy_r*wxy_r*wz_r*m.pi**1.5
    N=6.022e-1*C*vr
    g=1.0+(1-F_r+F_r*np.exp(-t/tf_r))/(1-F_r)/N/(1+4*D*t/wxy_r/wxy_r)/np.sqrt(1+4*D*t/wz_r/wz_r)
    corr_g=np.vstack((corr_g,g[:]))

    wxysq=wxy_r*wxy_r+wxy_b*wxy_b
    wzsq=wz_r*wz_r+wz_b*wz_b
    vcorr=2*wxy_b*wxy_b*wxy_r*wxy_r/wxysq*np.sqrt(2*wz_b*wz_b*wz_r*wz_r/wzsq)*m.pi**1.5
    N=6.022e-1*C*vcorr
    g=1+1/N/(1+8*D*t/wxysq)/np.sqrt(1+8*D*t/wzsq)*np.exp(-delta_z*delta_z/(8*D*t+wzsq))
    corr_g=np.vstack((corr_g,g))

    print "C=",C," D= ",D," dz= ",delta_z," V_blue= ",vb," V_red= ",vr, "V_cross=",vcorr

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()

# functions describing crosscorrelation functions that are derived from U1(z1)*exp(-(z2-z1)^2/s^2)*U2(z2)

def gc_noexp(eta,xi,t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz):
    return m.sqrt(m.pi)*k(eta-np.sqrt(D*t)*xi+dz,a1,R1,lambdaem1,n)*k(eta+np.sqrt(D*t)*xi,a2,R2,lambdaem2,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi+dz,w11,lambdaex1,n)+w2(eta+np.sqrt(D*t)*xi,w22,lambdaex2,n))

def gcz1_hermite(eta,t,D,w1,w2,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz):
    return np.sum(gc_noexp(eta,xh,t,D,w1,w2,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz)*yh)

def gc_hermite(t,D,w1,w2,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz):
    return quad(gcz1_hermite,0,200,args=(t,D,w1,w2,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz))

#Numerical Model Fitting Blue / Red / Cross Together
def g_oligo_all_n(b,t,data=None,sigma=None):

    wxy_b = b['w0_b'].value
    wxy_r = b['w0_r'].value
    lambdaex_b=b['lambdaex_b'].value
    lambdaem_b=b['lambdaem_b'].value
    lambdaex_r=b['lambdaex_r'].value
    lambdaem_r=b['lambdaem_r'].value
    a_b=b['a_b'].value
    a_r=b['a_r'].value
    r0_b=b['r0_b'].value
    r0_r=b['r0_r'].value
    D = b['D'].value
    n = b['n'].value
    C = 6.022e-1*b['C'].value
    dz=b['delta_z'].value

    #blue correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n)
    v2=vol2(wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n)
    vb=v1*v1/v2

    g=[1+g_hermite(tt,D,wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n)[0]/C/v1/v1 for tt in t]
#    gDb=np.array(gDb)
    corr_g=g[:]

    #red correlation
    v1=vol1(a_r,r0_r,lambdaem_r,n)
    v2=vol2(wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n)
    vr=v1*v1/v2

    g=[1.0+g_hermite(tt,D,wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n)[0]/C/v1/v1 for tt in t]
#    gDr =np.array(gDr)
    corr_g=np.vstack((corr_g,g[:]))

    #cross correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n)
    v2=vol1(a_r,r0_r,lambdaem_r,n)
    g=[1.0+gc_hermite(tt, D, wxy_b, wxy_r, a_b, a_r, r0_b, r0_r, lambdaex_b,lambdaem_b,lambdaex_r,lambdaem_r,n,dz)[0]/C/v1/v2 for tt in t]
#    gDc=np.array(gDc)
    corr_g=np.vstack((corr_g,g))

    print "C=",C/(6.022e-1)," D= ",D," dz= ",dz," V_blue= ",vb," V_red= ",vr, 'wxb=',wxy_b,'wxr=',wxy_r

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()

#Numerical Model Fitting Blue / Red / Cross Together
def g_oligo_all_nc(b,t,data=None,sigma=None):

    wxy_b = b['w0_b'].value
    wxy_r = b['w0_r'].value
    lambdaex_b=b['lambdaex_b'].value
    lambdaem_b=b['lambdaem_b'].value
    lambdaex_r=b['lambdaex_r'].value
    lambdaem_r=b['lambdaem_r'].value
    a_b=b['a_b'].value
    a_r=b['a_r'].value
    r0_b=b['r0_b'].value
    r0_r=b['r0_r'].value
    D = b['D'].value
    n = b['n'].value
    Cb = 6.022e-1*b['Cb'].value
    Cr = 6.022e-1*b['Cr'].value
    Cc = 6.022e-1*b['Cc'].value
    dz=b['delta_z'].value

    #blue correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n)
    v2=vol2(wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n)
    vb=v1*v1/v2

    g=[1+g_hermite(tt,D,wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n)[0]/Cb/v1/v1 for tt in t]
#    gDb=np.array(gDb)
    corr_g=g[:]

    #red correlation
    v1=vol1(a_r,r0_r,lambdaem_r,n)
    v2=vol2(wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n)
    vr=v1*v1/v2

    g=[1.0+g_hermite(tt,D,wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n)[0]/Cr/v1/v1 for tt in t]
#    gDr =np.array(gDr)
    corr_g=np.vstack((corr_g,g[:]))

    #cross correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n)
    v2=vol1(a_r,r0_r,lambdaem_r,n)
    g=[1.0+gc_hermite(tt, D, wxy_b, wxy_r, a_b, a_r, r0_b, r0_r, lambdaex_b,lambdaem_b,lambdaex_r,lambdaem_r,n,dz)[0]/Cc/v1/v2 for tt in t]
#    gDc=np.array(gDc)
    corr_g=np.vstack((corr_g,g))

    print "Cb=",Cb/(6.022e-1),"Cr=",Cr/(6.022e-1),"Cc=",Cc/(6.022e-1)," D= ",D," dz= ",dz," V_blue= ",vb," V_red= ",vr, 'wxb=',wxy_b,'wxr=',wxy_r

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
