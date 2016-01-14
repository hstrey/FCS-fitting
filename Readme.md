Python script for fluorescence correlation spectroscopy fitting

read_dilutions_dir:
script to read datasets at various concentrations from a data directory
it outputs for each concentration a csv file with the mean and standard deviation of the blue and red channel

read_oligos_dir:
same as read_dilutions_dir but for the oligo data

read_pcr_dir:
same as read_dilutions_dir but for the pcr data

FCS_Models, FCS_models_reversed:
Defines "lmfit" Models for FCS data fitting.
There are two main models:
1) Standard 3-d Gaussian model with or without triplet,
2) Fitting to a more realistic detection volume model that requires numerical calculation (slow but better).
In this model the user has the choice between two types of collection efficiency functions (k and k_real).

corr_average_model:
fits all data sets with all the models and outputs four csv files for the four different models

read_dilution_results:
displays an overview graph of the results obtained in corr_average_model

corr_average_all:
takes a selected subset of concentrations and fits them together to determine the parameter set (width, shape) of the molecular detection function.

fit_dilutions_with_para:
takes the parameter set that from corr_average_all and fits all dilution data with it.

model_fit_single, model_fit_single_rev:
fits a single data set to all the fitting models and displays the result

fit_oligos:
fits oligo data (single and cross-correlation) to a specific choice of model
