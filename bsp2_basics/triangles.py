from typing import List, Tuple

import numpy as np
import math

def define_triangle() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### 12326185 - matriculation number
    ### ABCDEFGH - matriculation number mapped to letters

    # P1 = (1 + C) −(1 + A) −(1 + E)
    P1 = np.array([1 + 3, -(1 + 1), -(1 + 6)])
    # P2 = −(1 + G) −(1 + B) (1 + H)
    P2 = np.array([-(1 + 8), -(1 + 2), 1 + 5])
    # P3 = −(1 + D) (1 + F ) −(1 + B)
    P3 = np.array([-(1 + 2), 1 + 1, -(1 + 2)])

    return P1, P2, P3

def define_triangle_vertices(P1:np.ndarray, P2:np.ndarray, P3:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # create the vectors P1P2 (from P1 to P2), P2P3 (from P2 to P3) and P3P1 (from P3 to P1)
    # = Spitze - Startpunkt
    P1P2 = P2 - P1
    P2P3 = P3 - P2
    P3P1 = P1 - P3

    return P1P2, P2P3, P3P1

def compute_lengths(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> List[float]: 
    # Berechne die Längen der Seiten des Dreiecks aka euklidische Norm
    # = Wurzel aus der Summe der Quadrate der einzelnen Vektorkomponenten aka (x, y, z)
    norms = []
    for side in [P1P2, P2P3, P3P1]:
        length = math.sqrt(side[0]**2 + side[1]**2 + side[2]**2)
        norms.append(length)

    return norms

def compute_normal_vector(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # normalvektor = kreuzprodukt 2er seitenvektoren
    # https://de.wikipedia.org/wiki/Normalenvektor#Normale_und_Normalenvektor_einer_Ebene
    n = np.cross(P1P2, P2P3)
    # normalvektor normalisieren
    # aus einem Normalenvektor wird ein Einheitsvektor, indem man ihn durch seine Länge teilt
    # np.linalg.norm(n) = euklidische Norm aka Länge des Vektors
    n_normalized = n / np.linalg.norm(n)

    return n, n_normalized

def compute_triangle_area(n:np.ndarray) -> float:
    # https://studyflix.de/mathematik/flacheninhalt-dreieck-vektoren-5664
    # dreiecksfläche = 1/2 * |n|
    # Hälfte des Betrages vom Kreuzprodukt 2er Seitenvektoren berechnest
    # Kreuzprodukt = Normalenvektor der Ebene
    # np.linalg.norm(n) = euklidische Norm aka Länge des Vektors aka Betrag
    area = 0.5 * np.linalg.norm(n)
    return area

def compute_angles(P1P2:np.ndarray,P2P3:np.ndarray,P3P1:np.ndarray) -> Tuple[float, float, float]:
    # berechung mit dem kosinussatz
    # https://de.wikipedia.org/wiki/Kosinussatz#Kosinussatz_f%C3%BCr_ebene_Dreiecke
    # Berechne die Längen der Seiten des Dreiecks
    a = np.linalg.norm(P2P3)
    b = np.linalg.norm(P3P1)
    c = np.linalg.norm(P1P2)
    
    # Berechne die Cosinus-Werte der Winkel mit dem Kosinussatz
    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
    
    # Wandle Cosinus-Werte in Winkel in Grad um
    alpha = np.arccos(cos_alpha) * (180 / math.pi)
    beta = np.arccos(cos_beta) * (180 / math.pi)
    gamma = np.arccos(cos_gamma) * (180 / math.pi)

    return alpha, beta, gamma

