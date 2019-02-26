#######################################################################################################################
# Function for creating spectral plot
#
# Author: Ethan Chen
# Date: 3 December 2018
# Email: ethanwc2@illinois.edu
#
#######################################################################################################################

import os
import matplotlib.pyplot as plt
plt.rcParams.update({'errorbar.capsize': 2})


def plot_spectra(spectra, stdev, filepath, number):

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.set_title("ROI #" + str(number))
    plt.ylim(0, 250)
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Intensity (A.U.)')
    y = spectra
    x = []
    for index in range(0, 36):
        x.append((2560+(index*20)))
    plt.errorbar(x, y, yerr=stdev, capthick=1)
    os.chdir(filepath)
    fig.savefig("Spectra" + str(number) + ".png")
    plt.close(fig)
    return


def plot_kmeans_spectra(spectra_data, kmeans_label):

    x = []
    y = []
    for index in range(0, 36):
        x.append((2560 + (index * 20)))
    fig, axes = plt.subplots(ncols=3, figsize=(32, 6))
    ax = axes.ravel()
    ax[0].set_title('Cluster 1 (Blue)')
    ax[1].set_title('Cluster 2 (Red)')
    ax[2].set_title('Cluster 3 (Yellow)')
    for a in ax:
        a.set_ylim(0, 200)
        a.set_xlabel('Wavenumber (cm^-1)')
        a.set_ylabel('Intensity (A.U.)')
    for index, label in enumerate(kmeans_label, start=0):
        y = spectra_data[index]
        if label == 0:
            ax[0].plot(x, y)
        elif label == 1:
            ax[1].plot(x, y)
        elif label == 2:
            ax[2].plot(x, y)
    plt.show()
