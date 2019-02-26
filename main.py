#######################################################################################################################
# Extracellular Vesicle Analysis with Coherent Anti-Stokes Raman Spectroscopy
#
# Author: Ethan Chen
# Date: 26 November 2018
# Email: ethanwc2@illinois.edu
#
#######################################################################################################################

import os
import numpy as np
import skimage as sk
from skimage import io, filters, morphology, segmentation, measure, color, feature, exposure
import matplotlib.pyplot as plt
import align_images as ai
import build_spectra as bs
import plot_spectra as ps
import pandas as pd
from sklearn.cluster import KMeans


def main():
    np.set_printoptions(threshold=np.nan)

    # filepath for CARS images
    filepath = "C:/Users/echen/PycharmProjects/ECVesicles/test_images"
    outpath = "C:/Users/echen/PycharmProjects/ECVesicles/output"

    im1 = io.imread("test_image.tif")
    # Note: TIF image may be 8 bit and needs to be scaled up to 16-bit (multiply by 257)
    im = im1*257

    # thresh_otsu = filters.threshold_otsu(im)
    thresh_yen = filters.threshold_yen(im, nbins=512)
    binary = im > thresh_yen

    bw = sk.morphology.closing(im > thresh_yen, sk.morphology.square(1))
    cleared = sk.segmentation.clear_border(bw)
    label_image = sk.measure.label(cleared)
    label_image_orig = label_image

    image_label_overlay = sk.color.label2rgb(label_image, image=im)

    # Image alignment of all CARS images using cross-correlation
    # http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    imagestack = ai.align_images(filepath)
    spectra_data = []
    flag = 1
    dfindex = []
    for index, region in enumerate(sk.measure.regionprops(label_image)):
        # take regions with approximate appropriate area
        if region.area <= 25:
            coordinates = region.coords
            spectra, stdev = bs.build_spectra(coordinates, imagestack)
            os.chdir(outpath)
            # Append spectra to master numpy array
            if flag == 1:
                spectra_data = spectra
                flag = 0
            else:
                spectra_data = np.vstack((spectra_data, spectra))
            ps.plot_spectra(spectra, stdev, outpath, region.label)
            dfindex.append(region.label)

            # Save plotted spectra data with labels
            # Build labeled spectra plot and save figure as an image

    # dfindex = np.arange(len(sk.measure.regionprops(label_image)))
    x = []
    for index in range(0, 36):
        x.append((2560 + (index * 20)))
    df = pd.DataFrame(spectra_data, index=dfindex, columns=x)

    # K-means clustering based on spectra and label in dataframe
    kmeans = KMeans(n_clusters=3, random_state=0).fit(spectra_data)
    df2 = pd.DataFrame(kmeans.labels_, index=dfindex, columns=['kmeans'])
    df = pd.concat([df, df2], axis=1, sort=False)
    df.to_csv('Data.csv')
    kmeans_label_image = label_image
    for index, label in enumerate(kmeans.labels_, start=0):
        kmeans_label_image[kmeans_label_image == dfindex[index]] = label+1
    kmeans_image_rgb = sk.color.label2rgb(kmeans_label_image, image=im, bg_label=0, colors=['blue', 'red', 'yellow'])

    # TODO: Compare spectra data of K-Means clusters using PCA
    # TODO: https://www.researchgate.net/post/What_is_the_best_way_to_compare_two_spectral_data_and_how_to_quantify

    # fig, axes = plt.subplots(ncols=1, figsize=(16, 6))
    # ax = axes.ravel()

    # ax[0].imshow(cleared, cmap='binary_r')
    # ax[0].set_title('Original image')

    # ax[1].imshow(kmeans_image_rgb)
    # ax[1].set_title('K-means Clustering')

    # for a in ax:
    #   a.axis('off')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("K-Means Clustering")
    ax.axis('off')
    ax.imshow(kmeans_image_rgb)
    # fig.savefig("K-Means.png")
    plt.show()
    ps.plot_kmeans_spectra(spectra_data, kmeans.labels_)


if __name__ == "__main__":
    main()
