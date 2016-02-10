import numpy as np
from lmfit import Model

#####################################################################
# 3-d Gaussian models
#####################################################################

# 3-d Gaussian focus FCS model with triplet correction
def g_FCS(t,C,wxy,wz,D,F,tf):
    v=wxy*wxy*wz*np.pi**1.5
    N=C*6.022e-1*v
    triplet=(1-F+F*np.exp(-t/tf))/(1-F)
    return 1.0+triplet/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)

# 3-d Gaussian focus FCS model
def g(t,C,wxy,wz,D):
    v=wxy*wxy*wz*np.pi**1.5
    N=C*6.022e-1*v
    return 1.0+1.0/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)

# 3-d Gaussian focus FCS model with triplet correction
def g_t(t,C,wxy,wz,D,F,tf):
    v=wxy*wxy*wz*np.pi**1.5
    N=C*6.022e-1*v
    triplet=(1-F+F*np.exp(-t/tf))/(1-F)
    return 1.0+triplet/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)

modelFCS_t = Model(g_t,independent_vars=['t'])
modelFCS = Model(g,independent_vars=['t'])

#####################################################################
# Combined fit dilutions residual functions
#####################################################################

def g_all(b,t,c,data=None,sigma=None):
    #C=b['C'].value
    corr_g=None
    D=b['D'].value
    wxy=b['wxy'].value
    wz=b['wz'].value
    slope=b['slope'].value
    for conc in c:
        C=slope*b[conc].value
        v=wxy*wxy*wz*np.pi**1.5
        N=C*6.022e-1*v
        g=1+1/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))

    print "wxy = ", wxy,"wz = ", wz,"slope",slope

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
    slope=b['slope'].value
    for conc in c:
        C=slope*b[conc].value
        v=wxy*wxy*wz*np.pi**1.5
        N=C*6.022e-1*v
        g=1+(1-F+F*np.exp(-t/tf))/(1-F)/N/(1+4*D*t/wxy/wxy)/np.sqrt(1+4*D*t/wz/wz)
        if corr_g is None:
            corr_g=g
        else:
            corr_g=np.vstack((corr_g,g))

    print "wxy = ", wxy, "wz = ", wz,"slope",slope

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()

