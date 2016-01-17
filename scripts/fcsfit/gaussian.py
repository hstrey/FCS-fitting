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
