# Copyright TU Wien (2022) - EVC: Task3
# Computer Vision Lab
# Institute of Computer Graphics and Algorithms

import numpy as np

def evc_white_balance(input_image: np.ndarray, white: np.ndarray) -> np.ndarray:
    """evc_white_balance performs white balancing manually.
    
      INPUT
      input_image ... image
      white       ... a color (as RGB vector) that should become the new white
    
      OUTPUT
      result      ... result after white balance"""
    
    
    # Überprüfe, ob der white-Vektor Nullen enthält
    # element-wise comparison of each element in the white array with zero
    # np.any() function tests whether any element in the input array is True
    white_is_zero = np.any(white == 0)

    # Erstelle eine Kopie des Eingabebildes, float weil das Ergebnis durch division float ist
    # sonst sind änderungen auch ausserhalb der Funktion sichtbar
    result = np.copy(input_image).astype(float)

    # Führe den Weißabgleich durch
    if white_is_zero:
        # Wenn der white-Vektor Nullen enthält, setze die entsprechenden Kanäle auf 1 um eine Division durch Null zu vermeiden
        white_nonzero = white.copy()
        white_nonzero[white_nonzero == 0] = 1
        result /= white_nonzero
    else:
        # Andernfalls führe die normale Skalierung durch
        result /= white
    return result
