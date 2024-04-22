# Copyright TU Wien (2022) - EVC: Task3
# Computer Vision Lab
# Institute of Computer Graphics and Algorithms

from typing import Tuple

import numpy as np

def evc_prepare_histogram_range(input_image: np.ndarray, low: float, high: float) -> Tuple[float, float]:
    """evc_prepare_histogram_range first calculates the new upper- and lower-
    bounds. During the normalization, those two values are then mapped to [0,1].
    If 'low' < 0, it should be set to 0.
    If 'high' > than the maximum intensity in the image, it should be set
    to the maximum intensity.

      INPUT
      input_image	... image
      low   		... current black value
      high  		... current white value

      OUTPUT
      newLow        ... new black value
      newHigh       ... new white value"""

    # Set newLow to 0 if low is less than 0
    newLow = max(low, 0.0)

    # Set newHigh to the maximum intensity in the image if high is greater
    max_intensity = np.max(input_image)
    newHigh = min(high, max_intensity)
    
    return newLow, newHigh


def evc_transform_histogram(input_image: np.ndarray, newLow: float, newHigh: float) -> np.ndarray:
    """ evc_transform_histogram performs the 'histogram normalization' and
        maps the interval [newLow, newHigh] to [0, 1].

        INPUT
        input_image ... image
        newLow   	... black value
        newHigh  	... white value

        OUTPUT
        result		... image after the histogram normalization"""

    # Subtract newLow from the input image
    adjusted_image = input_image.astype(float) - newLow

    # Clip negative values to 0
    adjusted_image[adjusted_image < 0] = 0

    # Normalize the adjusted image to the range [0, 1]
    result = adjusted_image / (newHigh - newLow)

    return result


def evc_clip_histogram(input_image: np.ndarray) -> np.ndarray:
    """ After the transformation of the histogram, evc_clip_histogram sets all
    values that are < 0 to 0 and values that are > 1 to 1.

      INPUT
      input_image   ... image after the histogram normalization

      OUTPUT
      result		... image after the clipping operation"""

    # Clip values less than 0 to 0
    result = np.clip(input_image, 0.0, None)

    # Clip values greater than 1 to 1
    result = np.clip(result, None, 1.0)
    
    return result
