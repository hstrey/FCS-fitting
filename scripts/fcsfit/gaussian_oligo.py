import numpy as np

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

    vb=wxy_b*wxy_b*wz_b*np.pi**1.5
    N=6.022e-1*C*vb
    g=1.0+(1-F_b+F_b*np.exp(-t/tf_b))/(1-F_b)/N/(1+4*D*t/wxy_b/wxy_b)/np.sqrt(1+4*D*t/wz_b/wz_b)
    corr_g=g[:]

    vr=wxy_r*wxy_r*wz_r*np.pi**1.5
    N=6.022e-1*C*vr
    g=1.0+(1-F_r+F_r*np.exp(-t/tf_r))/(1-F_r)/N/(1+4*D*t/wxy_r/wxy_r)/np.sqrt(1+4*D*t/wz_r/wz_r)
    corr_g=np.vstack((corr_g,g[:]))

    wxysq=wxy_r*wxy_r+wxy_b*wxy_b
    wzsq=wz_r*wz_r+wz_b*wz_b
    vcorr=2*wxy_b*wxy_b*wxy_r*wxy_r/wxysq*np.sqrt(2*wz_b*wz_b*wz_r*wz_r/wzsq)*np.pi**1.5
    N=6.022e-1*C*vcorr
    g=1+1/N/(1+8*D*t/wxysq)/np.sqrt(1+8*D*t/wzsq)*np.exp(-delta_z*delta_z/(8*D*t+wzsq))
    corr_g=np.vstack((corr_g,g))

    print "C=",C," D= ",D," dz= ",delta_z," V_blue= ",vb," V_red= ",vr, "V_cross=",vcorr

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()

def g_oligo_all_c(b,t,data=None,sigma=None):
    D=b['D'].value
    wxy_b=b['wxy_b'].value
    wz_b=b['wz_b'].value
    wxy_r=b['wxy_r'].value
    wz_r=b['wz_r'].value
    delta_z=b['delta_z'].value
    Cb = b['Cb'].value
    Cr = b['Cr'].value
    Cc = b['Cc'].value
    F_b=b['F_b'].value
    F_r=b['F_r'].value
    tf_b=b['tf_b'].value
    tf_r=b['tf_r'].value

    vb=wxy_b*wxy_b*wz_b*np.pi**1.5
    Nb=6.022e-1*Cb*vb
    g=1.0+(1-F_b+F_b*np.exp(-t/tf_b))/(1-F_b)/Nb/(1+4*D*t/wxy_b/wxy_b)/np.sqrt(1+4*D*t/wz_b/wz_b)
    corr_g=g[:]

    vr=wxy_r*wxy_r*wz_r*np.pi**1.5
    Nr=6.022e-1*Cr*vr
    g=1.0+(1-F_r+F_r*np.exp(-t/tf_r))/(1-F_r)/Nr/(1+4*D*t/wxy_r/wxy_r)/np.sqrt(1+4*D*t/wz_r/wz_r)
    corr_g=np.vstack((corr_g,g[:]))

    wxysq=wxy_r*wxy_r+wxy_b*wxy_b
    wzsq=wz_r*wz_r+wz_b*wz_b
    vcorr=2*wxy_b*wxy_b*wxy_r*wxy_r/wxysq*np.sqrt(2*wz_b*wz_b*wz_r*wz_r/wzsq)*np.pi**1.5
    Nc=6.022e-1*Cc*vcorr
    g=1+1/Nc/(1+8*D*t/wxysq)/np.sqrt(1+8*D*t/wzsq)*np.exp(-delta_z*delta_z/(8*D*t+wzsq))
    corr_g=np.vstack((corr_g,g))

    print "Cb=",Cb,"Cr=",Cr,"Cc=",Cc," D= ",D," dz= ",delta_z," V_blue= ",vb," V_red= ",vr, "V_cross=",vcorr

    if data is None:
        return corr_g
    corr_res=(corr_g-data)/sigma
    return corr_res.flatten()
