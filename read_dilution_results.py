import pandas as pd
import matplotlib.pyplot as plt

#from GaussianModels import modelFCS_t, modelFCS, modelFCS_n, modelFCS_nt, vol1, vol2, modelNoise, g_t, g_n, g, g_nt, noise

#defines the location of the data
datadir='../062415/50um/dilutions/'

#defines the columns of the output

data3dGaussian=pd.read_csv(datadir+'gaussian_B1R8_real_cVary.csv')
data3dGaussianTriplet=pd.read_csv(datadir+'gaussian_triplet_B1R8_real_cVary.csv')
dataNumerical=pd.read_csv(datadir+'Numerical_B1R8_real_cVary.csv')
dataNumericalTriplet=pd.read_csv(datadir+'NumericalTriplet_B1R8_real_cVary.csv')

red3d=data3dGaussian[data3dGaussian['color']=="R"]
blue3d=data3dGaussian[data3dGaussian['color']=="B"]
red3dt=data3dGaussianTriplet[data3dGaussianTriplet['color']=="R"]
blue3dt=data3dGaussianTriplet[data3dGaussianTriplet['color']=="B"]
redN=dataNumerical[dataNumerical['color']=="R"]
blueN=dataNumerical[dataNumerical['color']=="B"]
redNt=dataNumericalTriplet[dataNumericalTriplet['color']=="R"]
blueNt=dataNumericalTriplet[dataNumericalTriplet['color']=="B"]

plt.figure()
plt.subplot(6,2,1)
plt.errorbar(blue3d['C'],blue3d['wxy'],yerr=blue3d['wxy_stderr'],fmt="ob")
plt.errorbar(blue3dt['C'],blue3dt['wxy'],yerr=blue3dt['wxy_stderr'],fmt="sb")
plt.ylabel("wxy in microns")

plt.subplot(6,2,2)
plt.errorbar(red3d['C'],red3d['wxy'],yerr=red3d['wxy_stderr'],fmt="or")
plt.errorbar(red3dt['C'],red3dt['wxy'],yerr=red3dt['wxy_stderr'],fmt="sr")

plt.subplot(6,2,3)
plt.errorbar(blueN['C'],blueN['w0'],yerr=blueN['w0_stderr'],fmt="ob")
plt.errorbar(blueNt['C'],blueNt['w0'],yerr=blueNt['w0_stderr'],fmt="sb")
plt.ylabel("w0 in microns")

plt.subplot(6,2,4)
plt.errorbar(redN['C'],redN['w0'],yerr=redN['w0_stderr'],fmt="or")
plt.errorbar(redNt['C'],redNt['w0'],yerr=redNt['w0_stderr'],fmt="sr")

plt.subplot(6,2,5)
plt.errorbar(blue3d['C'],blue3d['wz'],yerr=blue3d['wz_stderr'],fmt="ob")
plt.errorbar(blue3dt['C'],blue3dt['wz'],yerr=blue3dt['wz_stderr'],fmt="sb")
plt.ylabel("wz in microns")

plt.subplot(6,2,6)
plt.errorbar(red3d['C'],red3d['wz'],yerr=red3d['wz_stderr'],fmt="or")
plt.errorbar(red3dt['C'],red3dt['wz'],yerr=red3dt['wz_stderr'],fmt="sr")

plt.subplot(6,2,7)
plt.errorbar(blueN['C'],blueN['r0'],yerr=blueN['r0_stderr'],fmt="ob")
plt.errorbar(blueNt['C'],blueNt['r0'],yerr=blueNt['r0_stderr'],fmt="sb")
plt.ylabel("r0 in microns")

plt.subplot(6,2,8)
plt.errorbar(redN['C'],redN['r0'],yerr=redN['r0_stderr'],fmt="or")
plt.errorbar(redNt['C'],redNt['r0'],yerr=redNt['r0_stderr'],fmt="sr")

plt.subplot(6,2,9)
plt.errorbar(blueNt['C'],blueNt['F'],yerr=blueNt['F_stderr'],fmt="sb")
plt.ylabel("fraction of triplets")

plt.subplot(6,2,10)
plt.errorbar(redNt['C'],redNt['F'],yerr=redNt['F_stderr'],fmt="sr")

plt.subplot(6,2,11)
plt.errorbar(blueNt['C'],blueNt['tf'],yerr=blueNt['tf_stderr'],fmt="sb")
plt.xlabel("concentration in molecules/micron^2")
plt.ylabel("triplet relaxation time in s")

plt.subplot(6,2,12)
plt.errorbar(redNt['C'],redNt['tf'],yerr=redNt['tf_stderr'],fmt="sr")
plt.xlabel("concentration in molecules/micron^2")

plt.show()
