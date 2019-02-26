#######################################################################################################################
# Extracellular Vesicle Analysis with Coherent Anti-Stokes Raman Spectroscopy
#
# Author: Ethan Chen
# Date: 26 November 2018
# Email: ethanwc2@illinois.edu
#
#######################################################################################################################

import pytest
import os
import numpy as np
import skimage as sk
from skimage import io, filters, morphology, segmentation, measure, color, feature
import matplotlib.pyplot as plt
import build_spectra as bs
import plot_spectra as ps
from scipy.ndimage import fourier_shift
from PIL import Image
import align_images as a
import build_spectra as bs
import plot_spectra as ps

filepath = "C:/Users/echen/PycharmProjects/ECVesicles/test_images"
outpath = "C:/Users/echen/PycharmProjects/ECVesicles/output"
rootpath = "C:/Users/echen/PycharmProjects/ECVesicles"


def set_up():
    np.set_printoptions(threshold=np.nan)

    im = io.imread("test_image.tif")
    # Note: TIF image may be 8 bit and needs to be scaled up to 16-bit (multiply by 257)
    im = im * 257

    # thresh_otsu = filters.threshold_otsu(im)
    thresh_yen = filters.threshold_yen(im)
    binary = im > thresh_yen

    bw = sk.morphology.closing(im > thresh_yen, sk.morphology.square(3))
    cleared = sk.segmentation.clear_border(bw)
    label_image = sk.measure.label(cleared)

    image_label_overlay = sk.color.label2rgb(label_image, image=im)

    for region in sk.measure.regionprops(label_image):
        # take regions with approximate appropriate area
        if region.area <= 15:
            coordinates = region.coords
            return coordinates


def clean_up():
    os.chdir(rootpath)


def test_segmentation():
    np.set_printoptions(threshold=np.nan)

    im = io.imread("test_image.tif")
    # Note: TIF image may be 8 bit and needs to be scaled up to 16-bit (multiply by 257)
    im = im * 257

    thresh_otsu = filters.threshold_otsu(im)
    # thresh_min = filters.threshold_minimum(im)
    thresh_adaptive = filters.threshold_adaptive(im, 101, offset=0)
    thresh_yen = filters.threshold_yen(im, nbins=512)
    binary = im > thresh_yen
    binary2 = im > thresh_otsu

    bw = sk.morphology.closing(im > thresh_yen, sk.morphology.square(1))
    cleared = sk.segmentation.clear_border(bw)
    label_image = sk.measure.label(cleared)

    image_label_overlay = sk.color.label2rgb(label_image, image=im, bg_label=0)

    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    ax = axes.ravel()

    ax[0].imshow(binary, cmap='binary_r')
    ax[0].set_title('Original image')

    ax[1].imshow(label_image, cmap='binary_r')
    ax[1].set_title('Segmentation')

    for a1 in ax:
        a1.axis('off')

    plt.show()


@pytest.mark.skip(reason="no way of currently testing this")
def test_align_images():
    pass
    os.chdir(rootpath)
    imagestack = a.align_images(filepath)
    # Output imagestack to saved files for manual checking of alignment
    # Change active directory to test output directory
    os.chdir("C:/Users/echen/PycharmProjects/ECVesicles/output")
    for counter, image in enumerate(imagestack):
        sk.io.imsave("Image" + str(counter) + ".tif", image)
    clean_up()


@pytest.mark.skip(reason="no way of currently testing this")
def test_build_spectra():
    pass
    coordinates = set_up()
    spectra = bs.build_spectra(coordinates, filepath)
    # print(spectra)
    clean_up()


@pytest.mark.skip(reason="no way of currently testing this")
def test_plot_spectra():
    pass
    coordinates = set_up()
    spectra = bs.build_spectra(coordinates, filepath)
    ps.plot_spectra(spectra, outpath, 1)
    clean_up()
