#######################################################################################################################
# Function for scraping raw spectral data from CARS images based on ROI
#
# Author: Ethan Chen
# Date: 26 November 2018
# Email: ethanwc2@illinois.edu
#
#######################################################################################################################

import os
import skimage as sk
from skimage import io
import numpy as np


def build_spectra(coordinates, imagestack):

    # Builds a spectra for a single ROI
    # Coordinates passed in to this function are a single set for one ROI
    spectra = []
    stdev = []

    # ############# DEPRECATED ##############################
    # filelist = []
    # for file in os.listdir(folderpath):
    #     if file.endswith(".tif"):
    #         filelist.append(file)
    # #######################################################

    for image in imagestack:
        intensity = 0
        calc_std = []

        # Calculate [mean?] intensity of ROI defined by coordinates
        # NOTE: Unsure if mean intensity is the appropriate value for the spectra here
        for index, coords in enumerate(coordinates, start=1):
            intensity += image[coords[0], coords[1]]
            calc_std.append(image[coords[0], coords[1]])

        intensity /= index
        stdev.append(np.std(calc_std))
        spectra.append(intensity)

    # Returns spectra as an array of y-values (intensity) with no x-values
    # Returns standard deviations of spectra
    return spectra, stdev


