

import os
import numpy as np
import skimage as sk
from skimage import feature
from scipy.ndimage import fourier_shift
from PIL import Image


def align_images(filepath):
    # Returns numpy array of aligned images

    os.chdir(filepath)
    filelist = []
    filecount = 0

    for file in os.listdir(filepath):
        if file.endswith(".tif"):
            filelist.append(file)
            filecount += 1

    # Load images into numpy array
    imagestack = np.array([np.array(Image.open(fname)) for fname in filelist])
    # print(filecount)

    for index, reference_image in enumerate(imagestack, start=0):
        # Check that reference is not the last image in stack
        if index+1 >= filecount:
            break

        test_image = imagestack[index+1]

        # Calculate shift
        shift, error, diffphase = sk.feature.register_translation(reference_image, test_image)
        shift[0] *= -1
        shift[1] *= -1

        # Apply shift to correct image
        # Might have to invert the shift
        offset_image = fourier_shift(np.fft.fftn(test_image), shift)
        corrected_image = np.fft.ifftn(offset_image)
        imagestack[index+1] = corrected_image

    return imagestack

