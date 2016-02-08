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

from fcsfit.common import k_real
from fcsfit.realistic_reversed import vol1,vol2
from fcsfit.realistic_real_reversed import vol1r, vol2r, g_hermite, g_hermite2

#defines the location of the data
datadir='../../data/dilutions/'

final_para=pd.read_csv(datadir+"final_parameters.csv")

print "w0: ",final_para['w0'][4]
print "r0: ",final_para['r0'][4]
print "D: ",final_para['D'][4]
print "a: ",final_para['a'][4]
print "lambdaex: ",final_para['lambdaex'][4]
print "lambdaem: ",final_para['lambdaem'][4]


print "vol1: ",vol1(final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaem'][4],
           1.33,k=k_real)

print "vol1r: ",vol1r(final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaem'][4],
           1.33)

v2 = vol2(final_para['w0'][4],
           final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaex'][4],
           final_para['lambdaem'][4],
           1.33,k=k_real)

print "v2: ",v2

v2r = vol2r(final_para['w0'][4],
           final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaex'][4],
           final_para['lambdaem'][4],
           1.33)
print "vol2r: ",v2r

gh = g_hermite(0.0,
            final_para['D'][4],
            final_para['w0'][4],
            final_para['a'][4],
            final_para['r0'][4],
            final_para['lambdaex'][4],
            final_para['lambdaem'][4],
            1.33)

print gh/v2
print gh/v2r

gh2 = g_hermite2(0.0,
            final_para['D'][4],
            final_para['w0'][4],
            final_para['a'][4],
            final_para['r0'][4],
            final_para['lambdaex'][4],
            final_para['lambdaem'][4],
            1.33,k=k_real)

print gh2/v2
print gh2/v2r
