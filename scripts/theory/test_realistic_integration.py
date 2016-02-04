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
from fcsfit.realistic_real_reversed import vol1r, vol2r

#defines the location of the data
datadir='../../data/dilutions/'

final_para=pd.read_csv(datadir+"final_parameters.csv")

print vol1(final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaem'][4],
           1.33,k=k_real)

print vol1r(final_para['a'][4],
           final_para['r0'][4],
           final_para['lambdaem'][4],
           1.33)
