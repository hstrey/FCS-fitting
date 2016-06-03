import numpy as np
from common import k,k_real,xh,yh,maxz,w2
from realistic_reversed import vol1,vol2,g_hermite
from lmfit import Model
from scipy.integrate import quad

def gc_noexp(eta,xi,t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k):
    return np.sqrt(np.pi)*k(eta-np.sqrt(D*t)*xi+dz/2.0,a1,R1,lambdaem1,n)*k(eta+np.sqrt(D*t)*xi-dz/2.0,a2,R2,lambdaem2,n)/(8*D*t+w2(eta-np.sqrt(D*t)*xi+dz/2.0,w11,lambdaex1,n)+w2(eta+np.sqrt(D*t)*xi-dz/2.0,w22,lambdaex2,n))

def gc_hermite(t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k):
    return np.sum([gcz1(x,t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k) for x in xh]*yh)

def gcz1(xi,t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k):
    return quad(gc_noexp,-maxz,maxz,args=(xi,t,D,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k))[0]

def g0c(z,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k):
    return k(z+dz/2.0,a1,R1,lambdaem1,n)*k(z-dz/2.0,a2,R2,lambdaem2,n)/np.sqrt(w2(z+dz/2.0,w11,lambdaex1,n))/np.sqrt(w2(z-dz/2.0,w22,lambdaex2,n))

def g0f(z,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k):
    return k(z+dz/2.0,a1,R1,lambdaem1,n)*k(z-dz/2.0,a2,R2,lambdaem2,n)/np.sqrt(w2(z,w11,lambdaex1,n))/np.sqrt(w2(z,w22,lambdaex2,n))

def vol2c(w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k=k):
    return np.pi/4.0*quad(g0c,-maxz-10.0,maxz+10.0,args=(w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k))[0]

def vol2f(w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k=k):
    return np.pi/4.0*quad(g0f,-maxz-10.0,maxz+10.0,args=(w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,k))[0]

def g_nc(t,D,C,w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,cef=k):

    v1=vol1(a1,R1,lambdaem1,n,cef=cef)
    v2=vol1(a2,R2,lambdaem2,n,cef=cef)

    print "w1 = ",w11,"w2 = ",w22,"R1 = ",R1,"R2 = ",R2,"c = ",C, "vol1 = ",v1,"vol 2 = ",v2

    return 1.0+np.array([gc_hermite(tt, D, w11,w22,a1,a2,R1,R2,lambdaex1,lambdaem1,lambdaex2,lambdaem2,n,dz,cef)[0] for tt in t])/C/6.022e-1/v1/v2/2.0

modelFCS_nc = Model(g_nc,independent_vars=['t'])
modelFCS_nrc = Model(g_nc,independent_vars=['t'],k=k_real)

#Numerical Model Fitting Blue / Red / Cross Together
def g_oligo_all_n(b,t,data=None,sigma=None,cef=k):

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
    F_b=b['F_b'].value
    F_r=b['F_r'].value
    tf_b=b['tf_b'].value
    tf_r=b['tf_r'].value

    #blue correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n,cef=cef)
    v2=vol2(wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n,cef=cef)
    vb=v1*v1/v2

    g=1.0+np.array([(1-F_b+F_b*np.exp(-tt/tf_b))/(1-F_b)*g_hermite(tt,D,wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n,k) for tt in t])/C/v1/v1
#    gDb=np.array(gDb)
    corr_g=g[:]

    #red correlation
    v1=vol1(a_r,r0_r,lambdaem_r,n,cef=cef)
    v2=vol2(wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n,cef=cef)
    vr=v1*v1/v2

    g=1.0+np.array([(1-F_r+F_r*np.exp(-tt/tf_r))/(1-F_b)*g_hermite(tt,D,wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n,cef) for tt in t])/C/v1/v1
#    gDr =np.array(gDr)
    corr_g=np.vstack((corr_g,g[:]))

    #cross correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n,cef=cef)
    v2=vol1(a_r,r0_r,lambdaem_r,n,cef=cef)
    g=1.0+np.array([gc_hermite(tt, D, wxy_b, wxy_r, a_b, a_r, r0_b, r0_r, lambdaex_b,lambdaem_b,lambdaex_r,lambdaem_r,n,dz,cef) for tt in t])/C/v1/v2/2.0
#    gDc=np.array(gDc)
    corr_g=np.vstack((corr_g,g))

    print "C=",C/(6.022e-1)," D= ",D," dz= ",dz," V_blue= ",vb," V_red= ",vr, 'w0b=',wxy_b,'w0r=',wxy_r

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()

#Numerical Model Fitting Blue / Red / Cross Together
def g_oligo_all_nc(b,t,data=None,sigma=None,cef=k):

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
    v1=vol1(a_b,r0_b,lambdaem_b,n,cef=cef)
    v2=vol2(wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n,cef=cef)
    vb=v1*v1/v2

    g=1.0+np.array([g_hermite(tt,D,wxy_b,a_b,r0_b,lambdaex_b,lambdaem_b,n,k) for tt in t])/Cb/v1/v1
#    gDb=np.array(gDb)
    corr_g=g[:]

    #red correlation
    v1=vol1(a_r,r0_r,lambdaem_r,n,cef=cef)
    v2=vol2(wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n,cef=cef)
    vr=v1*v1/v2

    g=1.0+np.array([g_hermite(tt,D,wxy_r,a_r,r0_r,lambdaex_r,lambdaem_r,n,cef) for tt in t])/Cr/v1/v1
#    gDr =np.array(gDr)
    corr_g=np.vstack((corr_g,g[:]))

    #cross correlation
    v1=vol1(a_b,r0_b,lambdaem_b,n,cef=cef)
    v2=vol1(a_r,r0_r,lambdaem_r,n,cef=cef)
    g=1.0+np.array([gc_hermite(tt, D, wxy_b, wxy_r, a_b, a_r, r0_b, r0_r, lambdaex_b,lambdaem_b,lambdaex_r,lambdaem_r,n,dz,cef) for tt in t])/Cc/v1/v2/2.0
#    gDc=np.array(gDc)
    corr_g=np.vstack((corr_g,g))

    print "Cb=",Cb/(6.022e-1),"Cr=",Cr/(6.022e-1),"Cc=",Cc/(6.022e-1)," D= ",D," dz= ",dz," V_blue= ",vb," V_red= ",vr, 'w0b=',wxy_b,'w0r=',wxy_r

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
