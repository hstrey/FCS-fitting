import numpy as np
from lmfit import Parameter, Parameters
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import time
import sys,os

# make sure that this scripts can find the fcsfit folder that is one level below
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from fcsfit.common import k_real,k
from fcsfit.realistic_reversed import g_n_norm

#defines the location of the data
datadir='../../data/dilutions/'

fit_dilutions=pd.read_csv(datadir+"fit_dilutions_plots.csv")
final_para=pd.read_csv(datadir+"final_parameters.csv")

# get experimental time range
t=np.append([0.0],np.array(fit_dilutions['tR']))

print "t[0]: ",t[0]

datadict={}
datadict['t']=t

gn=g_n_norm(t,final_para['D'][4],
                final_para['w0'][4],
                final_para['a'][4],
                final_para['r0'][4],
                final_para['lambdaex'][4],
                final_para['lambdaem'][4],
                1.33,cef=k)

print "gn(0): ",gn[0]

datadict['gn']=gn

gnr=g_n_norm(t,final_para['D'][4],
                final_para['w0'][4],
                final_para['a'][4],
                final_para['r0'][4],
                final_para['lambdaex'][4],
                final_para['lambdaem'][4],
                1.33, cef=k_real)

print "gnr(0): ",gnr[0]

datadict['gnr']=gnr

# data=pd.DataFrame(datadict)
# data.to_csv(datadir+'integration_limit_rev_10_32.csv')
