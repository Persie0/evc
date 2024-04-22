import numpy as np
import scipy.ndimage
from PIL import Image

import utils


def read_img(inp:str) -> Image.Image:
    """
        Returns a PIL Image given by its input path.
    """
    img =  Image.open(inp)
    return img

def convert(img:Image.Image) -> np.ndarray:
    """
        Converts a PIL image [0,255] to a numpy array [0,1].
    """
    # Konvertierung des Bildes in ein Numpy-Array
    out = np.array(img, dtype=np.float32)
    
    # Skalierung der Pixelwerte (0...255) auf den Bereich [0, 1]
    out /= 255.0
    return out

def switch_channels(img:np.ndarray) -> np.ndarray:
    """
        Swaps the red and green channel of a RGB image given by a numpy array.
    """
    # Kopiere das Eingabebild, um das Ergebnisbild zu initialisieren
    out = np.copy(img)

    # (height, width, channels) -> (zeilen, spalten, kanäle)
    for y in range(img.shape[0]):  # über alle Zeilen
        for x in range(img.shape[1]):  # über alle Spalten
            # Lese die Werte der Rot- und Grünkanäle des aktuellen Pixels
            # (y-Koordinate, x-Koordinate, Kanal), wobei 0=Rot, 1=Grün, 2=Blau
            red_channel_value = img[y, x, 0]
            green_channel_value = img[y, x, 1]

            # Tausche die Werte der Rot- und Grünkanäle
            out[y, x, 0] = green_channel_value
            out[y, x, 1] = red_channel_value

    # kürzer form nach nächster aufgabe
    # out[:,:,0], out[:,:,1] = img[:,:,1], img[:,:,0]

    return out

def image_mark_green(img:np.ndarray) -> np.ndarray:
    """
        returns a numpy-array (HxW) with 1 where the green channel of the input image is greater or equal than 0.7, otherwise zero.
    """
    # : -> alle zeilen, : -> alle spalten, 1 -> grün
    # selektiert alle elemente der height und width dimension, aber nur den grünkanal
    # alle pixel mit grünwert >= 0.7 werden zu 1, sonst zu 0
    mask = np.array(img[:,:,1] >= 0.7, dtype=np.int8)

    return mask


def image_masked(img:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
        sets the pixels of the input image to zero where the mask is 1.
    """
        # Erstelle eine Kopie des Eingabebildes, um das Ergebnisbild zu initialisieren
    out = np.copy(img)

    # Setze die Pixel auf Schwarz, wo die Maske 1 ist
    # kann man machen wenn out und mask die gleiche shape haben
    # mask == 1 selektiert alle pixel, die in der maske 1 sind
    # out[mask == 1] selektiert alle pixel in out, die in der maske 1 sind
    out[mask == 1] = 0

    return out

def grayscale(img:np.ndarray) -> np.ndarray:
    """
        Returns a grayscale image of the input. Use utils.rgb2gray().
    """
    # benutzt bereitgestellte funktion um das bild zu konvertieren
    return utils.rgb2gray(img)

def cut_and_reshape(img_gray:np.ndarray) -> np.ndarray:
    """
        Cuts the image in half (x-dim) and stacks it together in y-dim.
    """
    # Zerschneide das Bild in der Mitte
    # (Zeilen, Spalten) -> (Höhe, Breite)
    # cut_index = hälfte der breite
    # // -> integer division aka floor division
    cut_index = img_gray.shape[1] // 2
    # [:, :cut_index] -> alle zeilen, bis zur hälfte der breite
    left_half = img_gray[:, :cut_index]
    # [:, cut_index:] -> alle zeilen, ab der hälfte der breite
    right_half = img_gray[:, cut_index:]

    # Füge den linken Teil unten an den rechten Teil an
    # vstack -> vertikales stacken, oben right_half, unten left_half
    out = np.vstack((right_half, left_half))

    return out

def filter_image(img:np.ndarray) -> np.ndarray:
    """
        filters the image with the gaussian kernel given below. 
        https://developer.nvidia.com/discover/convolution
        !!!missing removing the padding
    """
    gaussian = utils.gauss_filter(5, 2)
    
    # because the kernel is 5x5, we need to pad 2 pixels on each side, see https://developer.nvidia.com/discover/convolution
    # ((before_height, after_height), (before_width, after_width), (before_channels, after_channels))
    pad_width = ((2, 2), (2, 2), (0, 0))
    # padding with constant zeros
    out = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    height = img.shape[0]
    width = img.shape[1]

    # Convolve the image with the gaussian kernel
    # do not loop over padding pixels
    for y in range(2, height-2):
        for x in range(2, width-2):
            # run over 5x5 grid from current pixel and calculate the convolution
            for f_y in range(-2, 3):
                for f_x in range(-2, 3):
                    # add the convolution result to the output image
                    # see https://developer.nvidia.com/discover/convolution
                    out[y, x] += img[y+f_y, x+f_x] * gaussian[f_y+2, f_x+2]
    return out

def horizontal_edges(img:np.ndarray) -> np.ndarray:
    """
        Defines a sobel kernel to extract horizontal edges and convolves the image with it.
        ??? missing: removing the padding
    """
    # Sobel kernel for horizontal edges, as defined in the task
    kernel = np.array([[1, 2, 1], 
                       [0, 0, 0], 
                       [-1, -2, -1]])
    
    # Perform 2D correlation with the kernel =~ convolution
    # input -> image, weights -> kernel, mode -> padding mode, cval -> constant value for padding
    out = scipy.ndimage.correlate(input=img, weights=kernel, mode='constant', cval=0.0)

    return out
