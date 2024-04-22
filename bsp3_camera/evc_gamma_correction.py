# Copyright TU Wien (2022) - EVC: Task3
# Computer Vision Lab
# Institute of Computer Graphics and Algorithms

import numpy as np

def rgb2gray(rgb : np.ndarray):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.clip(gray, 0, 1)

def evc_compute_brightness(input_image: np.ndarray) -> np.ndarray:
    """evc_compute_brightness calculates the brightness of the input image.
       First the image is normalized by multiplying it with the reciprocal of
       the maximum value of all three color channels. The brightness is then
       retrieved by computing a gray-scale image. Afterwards the result
       is multiplied by the maximum value.
    
      INPUT
      input_image ... image matrix of dimension: (n, m, 3)
    
      OUTPUT
      brightness  ... brightness of the image, matrix of dimension (n, m)"""

        # Berechne den maximalen Wert über alle Farbkanäle
    max_value = np.max(input_image)

    # Normalisiere das Eingabebild
    normalized_image = input_image / max_value

    # Berechne die Helligkeitswerte mit rgb2gray
    brightness = rgb2gray(normalized_image)

    # Multipliziere das Ergebnis mit dem maximalen Wert
    brightness *= max_value

    return brightness

def evc_compute_chromaticity(input_image: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """ evc_compute_chromaticity calculates the chromaticity of the 'input' image
    using the 'brightness' values. Therefore the color channels of the input
    image are individually divided by the brightness values.
    
      INPUT
      input_image   ... image, dimension (n, m, 3)
      brightness    ... brightness values, dimension (n, m)

      OUTPUT
      chromaticity  ... chromaticity of the image, dimension (n, m, 3)"""

    # Berechne die Chromatizität, indem die Farbkanäle durch die Helligkeitswerte dividiert werden
    # 3te Dimension (Farbkanäle) wird aufgeteilt
    r, g, b = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
    r_chromaticity = r / brightness
    g_chromaticity = g / brightness
    b_chromaticity = b / brightness

    # Füge die Chromatizitätskanäle zu einem 3D-Array zusammen, depth (farbkanäle) stack (dstack)
    chromaticity = np.dstack((r_chromaticity, g_chromaticity, b_chromaticity))

    return chromaticity

def evc_gamma_correct(input_image: np.ndarray, gamma: float) -> np.ndarray:
    """evc_gamma_correct performs gamma correction on the 'input_image' image.
    This is done by raising it to the power of the reciprocal value of gamma
    (gamma**(-1)).
    
      INPUT
      input_image ... image
      gamma       ... gamma value
    
      OUTPUT
      corrected   ... image after gamma correction"""

    # Behandle den Fall gamma == 0 separat
    if gamma == 0:
        gamma = 0.0000000001  # Setze gamma auf einen sehr kleinen Wert

    # Führe die Gammakorrektur durch
    invGamma = 1.0 / gamma
    # potenzieren mit invGamma
    corrected = input_image ** invGamma

    return corrected

def evc_reconstruct(brightness_corrected: np.ndarray, chromaticity) -> np.ndarray:
    """ evc_reconstruct reconstructs the color values by multiplying the corrected
    brightness with the chromaticity.
    
      INPUT
      brightness_corrected  ... gamma-corrected brightness values
      chromaticity          ... chromaticity
    
      OUTPUT
      result                ... reconstructed image"""

    # Rekonstruiere die Farbwerte, indem die korrigierten Helligkeitswerte
    # mit der Chromatizität multipliziert werden
    r_recon = brightness_corrected * chromaticity[:, :, 0]
    g_recon = brightness_corrected * chromaticity[:, :, 1]
    b_recon = brightness_corrected * chromaticity[:, :, 2]

    # Füge die Farbkanäle zu einem 3D-Array zusammen
    result = np.dstack((r_recon, g_recon, b_recon))

    return result
