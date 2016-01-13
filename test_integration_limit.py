import numpy as np
from lmfit import Parameter, Parameters
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import time
import sys

from FCS_Models_reversed import g_n_norm ,k_real

#defines the location of the data
datadir='../062415/50um/dilutions/'

fit_dilutions=pd.read_csv(datadir+"fit_dilutions_plots.csv")
final_para=pd.read_csv(datadir+"final_parameters.csv")

# get experimental time range
t=np.array(fit_dilutions['tR'])

datadict={}
datadict['t']=t

gn=g_n_norm(t,final_para['D'][4],
                final_para['w0'][4],
                final_para['a'][4],
                final_para['r0'][4],
                final_para['lambdaex'][4],
                final_para['lambdaem'][4],
                1.33)-1.0

datadict['gn']=gn

gnr=g_n_norm(t,final_para['D'][4],
                final_para['w0'][4],
                final_para['a'][4],
                final_para['r0'][4],
                final_para['lambdaex'][4],
                final_para['lambdaem'][4],
                1.33, k=k_real)-1.0

datadict['gnr']=gnr

data=pd.DataFrame(datadict)
data.to_csv(datadir+'integration_limit_rev_10_16.csv')
