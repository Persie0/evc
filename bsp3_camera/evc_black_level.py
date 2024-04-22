# Copyright TU Wien (2022) - EVC: Task3
# Computer Vision Lab
# Institute of Computer Graphics and Algorithms

from typing import Tuple

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS


def evc_read_file_info(filename: str) -> Tuple[int, Tuple]:
    """evc_read_file_info extracts the black level (blackLevel) and the neutral
       white value (asShotNeutral) from the image file specified by filename.
    
      INPUT
      filename      ... filename of the image
    
      OUTPUT
      blackLevel    ... black level, which is stored in the image infos (pay attention to the typehint -> it should be an integer!)
      asShotNeutral ... neutral white value, which is stored in the image"""

    # Open the image file
    img = Image.open(filename)

    #TAGS is a dictionary that maps tag IDs to their respective names in the TIFF image format.
    #loop through all keys in img.tag_v2 and create a dictionary where the
    #keys are looked up in the TAGS dictionary based on keys from img.tag_v2
    #and the values are corresponding values from img.tag.
    meta_dict = {TAGS[key]: img.tag[key] for key in img.tag_v2}

    blackLevel = meta_dict["BlackLevel"][0]
    asShotNeutral = meta_dict["AsShotNeutral"]


    return blackLevel, asShotNeutral
    
def evc_transform_colors(input_image: np.ndarray, blackLevel: float) -> np.ndarray:
    """evc_transform_colors adjusts the contrast such that black (blackLevel and
    values below) becomes 0 and white becomes 1.
    The white value of the input image is 65535.
    
      INPUT
      input_image   ... input image
      blackLevel    ... black level of the input image
    
      OUTPUT
      result        ... image in double format where all values are
                        transformed from the interval [blackLevel, 65535]
                        to [0, 1]. All values below the black level have to
                        be 0."""
    
    # Subtract the blackLevel from the input image, float so that result is float
    adjusted_image = input_image.astype(float) - blackLevel

    # Clip negative values to 0
    adjusted_image[adjusted_image < 0] = 0

    # Normalize the adjusted image to the range [0, 1]
    # - blackLevel because we already subtracted it
    result = adjusted_image / (65535.0 - blackLevel)
    
    return result
