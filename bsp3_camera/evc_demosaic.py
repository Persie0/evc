# Copyright TU Wien (2022) - EVC: Task3
# Computer Vision Lab
# Institute of Computer Graphics and Algorithms

from typing import Tuple

import numpy as np
import numpy.matlib as matlib
import scipy.ndimage


def evc_demosaic_pattern(input_image: np.ndarray, pattern = 'RGGB') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ evc_demosaic_pattern extracts the red, green and blue values of the
     'input' image. Results are stored in the R, G, B variables.
    
      INPUT
      input_image   ... Bayer-Pattern image
      pattern       ... Bayer-Pattern

      OUTPUT
      R             ... red channel of the image (without interpolation)
      G             ... green channel of the image (without interpolation)
      B             ... blue channel of the image (without interpolation)"""

    height, width = input_image.shape
    R = np.zeros((height, width), dtype=input_image.dtype)
    G = np.zeros((height, width), dtype=input_image.dtype)
    B = np.zeros((height, width), dtype=input_image.dtype)

    if pattern == 'RGGB':
        # (jede zweite Zeile, jede zweite Spalte), beginnend bei 0,0
        R[::2, ::2] = input_image[::2, ::2]
        # (jede zweite Zeile, jede ungerade Spalte), beginnend bei 0,1
        G[::2, 1::2] = input_image[::2, 1::2]
        # (jede ungerade Zeile, jede zweite Spalte), beginnend bei 1,0
        G[1::2, ::2] = input_image[1::2, ::2]
        # (jede ungerade Zeile, jede ungerade Spalte), beginnend bei 1,1
        B[1::2, 1::2] = input_image[1::2, 1::2]
    elif pattern == 'BGGR':
        B[::2, ::2] = input_image[::2, ::2]
        G[::2, 1::2] = input_image[::2, 1::2]
        G[1::2, ::2] = input_image[1::2, ::2]
        R[1::2, 1::2] = input_image[1::2, 1::2]
    elif pattern == 'GRBG':
        G[::2, ::2] = input_image[::2, ::2]
        R[::2, 1::2] = input_image[::2, 1::2]
        B[1::2, ::2] = input_image[1::2, ::2]
        G[1::2, 1::2] = input_image[1::2, 1::2]
    elif pattern == 'GBRG':
        G[::2, ::2] = input_image[::2, ::2]
        B[::2, 1::2] = input_image[::2, 1::2]
        R[1::2, ::2] = input_image[1::2, ::2]
        G[1::2, 1::2] = input_image[1::2, 1::2]

    return R, G, B

def evc_transform_neutral(R: np.ndarray, G: np.ndarray, B: np.ndarray, asShotNeutral: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """evc_transform_neutral changes the red, green and blue channels depending
       on the neutral white value (asShotNeutral). Therefore every channel needs
       to be divided by the respective channel of the white value.
    
      INPUT
      R             ... red channel of the image
      G             ... green channel of the image
      B             ... blue channel of the image
      asShotNeutral ... neutral white value (RGB vector)
    
      OUTPUT
      R_trans       ... red channel of the image (changed by neutral white value)
      G_trans       ... green channel of the image (changed by neutral white value)
      B_trans       ... blue channel of the image (changed by neutral white value)"""

    # get r g b values of asShotNeutral
    # asShotNeutral = [asShotNeutral_R, asShotNeutral_G, asShotNeutral_B] values
    # = neutral white value as RGB vector
    asShotNeutral_R = asShotNeutral[0]
    asShotNeutral_G = asShotNeutral[1]
    asShotNeutral_B = asShotNeutral[2]

    # Perform white balance adjustment
    R_trans = R / asShotNeutral_R
    G_trans = G / asShotNeutral_G
    B_trans = B / asShotNeutral_B


    return R_trans, G_trans, B_trans

def evc_interpolate(red : np.ndarray, green : np.ndarray, blue : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """evc_interpolate interpolates the red, green and blue channels. In the
       final image, every pixel now has red, green and blue values.
        
        INPUT
        red         ... red channel of the image
        green       ... green channel of the image
        blue        ... blue channel of the image
        
        OUTPUT
        R_inter     ... red channel of the image (without missing values)
        G_inter     ... green channel of the image (without missing values)
        B_inter     ... blue channel of the image (without missing values)"""

    # Define the filter for the Green channel
    green_filter = np.array([[0.0, 0.25, 0.0],
                             [0.25, 1.0, 0.25],
                             [0.0, 0.25, 0.0]])
    
    # Apply the filter to the Green channel
    G_inter = scipy.ndimage.correlate(green, green_filter, mode='constant', cval=0.0)
    
    # For Red and Blue channels
    # if has only 2 neighbors (Cx or Cy) horizontally/vertically, use the average of the neighbors (upper and lower pixel or left and right pixel)
    # if has 4 neighbors (Cm), use the average of the diagonal neighbors (upper left, upper right, lower left, lower right)
    # berechneter Wert = mittelwert der 2 oder 4 Nachbarn
    red_blue_filter = np.array([[0.25, 0.5, 0.25],
                                [0.5, 1.0, 0.5],
                                [0.25, 0.5, 0.25]])
    
    # Apply the filter to the Red and Blue channels
    R_inter = scipy.ndimage.correlate(red, red_blue_filter, mode='constant', cval=0.0)
    B_inter = scipy.ndimage.correlate(blue, red_blue_filter, mode='constant', cval=0.0)
    
    return R_inter, G_inter, B_inter

def evc_concat(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """evc_concat combines the three individual red, green and blue channels to a
    single image.
    
      INPUT
      R             ... red channel of the image
      G             ... green channel of the image
      B             ... blue channel of the image
    
      OUTPUT
      result        ... resulting image"""

    # Stack the red, green, and blue channels along the third dimension to form the RGB image
    # axis=-1 because we want to stack the channels along the last dimension of the 3D array
    # axis=0 would stack the channels along the first dimension (height), axis=1 along the second dimension (width)
    # axis=-1 is equivalent to axis=2, which are the color channels aka depth
    result = np.stack((R, G, B), axis=-1)

    return result
