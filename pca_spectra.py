#######################################################################################################################
# Function for running PCA on a spectra
#
# Author: Ethan Chen
# Date: 16 February 2019
# Email: ethanwc2@illinois.edu
#
#######################################################################################################################

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA


def pca_spectra(spectra):
    pca = sklearnPCA(n_components=2)
    # TODO: Standardize/normalize data values prior to running PCA?
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    Y_sklearn = pca.fit_transform(spectra)
