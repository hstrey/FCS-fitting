Python script for fluorescence correlation spectroscopy fitting

read_dilutions_dir:
script to read datasets at various concentrations from a data directory
it outputs for each concentration a csv file with the mean and standard deviation of the blue and red channel

GaussianModels:
Defines "lmfit" Models for FCS data fitting.  There are two main models: 1) Standard 3-d Gaussian model with or without triplet, 2) Fitting to a more realistic detection volume model that requires numerical calculation (slow but better).

corr_average_model:
fits all data sets with all the models and outputs four csv files for the four different models

read_dilution_results:
displays an overview graph of the results obtained in corr_average_model
