from typing import List

import numpy as np
import matplotlib.pyplot as plt

def define_transformations() -> List[np.ndarray]:
    """
        Returns the four transformations t1, .., t4 to transform the quadrat. 
        The transformations are determined by using mscale, mrotate and mtranslate.

        Wir benutzen 3 dimensionen, da damit alle transformationen 
        (rotation, translation, scaling) in einer matrix dargestellt werden können.
    """
    ### rtl, rechts 1. angewandnte operation
    # 1. rotiere all vertices um 55 grad gegen den uhrzeigersinn (referenzpunkt ist der ursprung (0,0)
    # 2. translate rotierte vertices um -3 in x richtung
    t1 = mtranslate(-3, 0) @ mrotate(55)

    # 1. translate um -3 in x richtung
    # 2. rotiere um 55 grad gegen den uhrzeigersinn (referenzpunkt ist der ursprung (0,0))
    t2 = mrotate(55) @ mtranslate(-3, 0)

    # 1. skalierung um 3 in x und 2 in y richtung um l=3 in x und h=2 Rechteck zu erhalten
    # 2. rotiere um 70 grad gegen den uhrzeigersinn (referenzpunkt ist der ursprung (0,0)) -> danach sonst Probleme
    # 3. translate um 1 in x und 1 in y richtung
    t3 = mtranslate(3, 1 ) @ mrotate(70) @ mscale(3, 2)

    # 1. rotiere um 45 grad gegen den uhrzeigersinn (referenzpunkt ist der ursprung (0,0))
    #    , ergibt auf spitze stehendes quadrat
    # 2. skalierung um 1 in x und 3 in y richtung, auf spitze stehendes rechteck 
    #    wird auf y achse gestreckt
    t4 = mscale(1, 3) @ mrotate(45)

    return [t1, t2, t3, t4]

def mscale(sx : float, sy : float) -> np.ndarray:
    """
        Defines a scale matrix. The scales are determined by sx in x and sy in y dimension.
    """
    # siehe Homogeneous Coordinates (2) in Tranformations ppt

    m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    return m

def mrotate(angle : float) -> np.ndarray:
    """
        Defines a rotation matrix (z-axis) determined by the angle in degree (!).
    """
    # siehe Homogeneous Coordinates (2) in Tranformations ppt
    # umrechnung von grad in rad, da numpy cos und sin radian erwartet
    angle_rad = np.deg2rad(angle)
    m = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])

    return m
    
def mtranslate(tx : float, ty : float) -> np.ndarray:
    """
        Defines a translation matrix. tx in x, ty in y direction.
    """
    # siehe Homogeneous Coordinates (2) in Tranformations ppt

    m = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    return m

def transform_vertices(v : np.ndarray, m : np.ndarray) -> np.ndarray:
    """
        transform the (3xN) vertices given by v with the (3x3) transformation matrix determined by m.
    """
    # siehe Homogeneous Coordinates (2) in Tranformations ppt
    # m ist die transformationsmatrix
    # v ist die matrix mit den vertices
    # die transformation ist dann m * v
    # @ ist das symbol für matrix multiplikation
    out = m @ v

    return out

def display_vertices(v : np.ndarray, title : str) -> None:
    """
        Plot the vertices in a matplotlib figure.
    """
    # create the figure and set the title
    plt.figure()
    plt.axis('square')

    plt.title(title)

    # x and y limits
    plt.xlim((-6,6))
    plt.ylim((-6,6))
    plt.xticks(range(-6,6))
    plt.yticks(range(-6,6))

    # plot coordinate axis
    plt.axvline(color='black')
    plt.axhline(color='black')
    plt.grid()
    
    # we just add the last element, so plot can do our job :)
    v_ = np.concatenate((v, v[:, 0].reshape(3,-1)), axis=1)

    plt.plot(v_[0, :], v_[1, :], linewidth=3)
    plt.show()
