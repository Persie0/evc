from typing import Tuple
import numpy as np
    
def define_structures() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Defines the two vectors v1 and v2 as well as the matrix M determined by your matriculation number.
    """
    ### 12326185 - matriculation number
    ### ABCDEFGH - matriculation number mapped to letters

    #DAC
    v1 = np.array([2, 1, 3])
    #FBE
    v2 = np.array([1, 2, 6])
    #DBC, BGA, EHF
    M = np.array([[2, 2, 3], [2, 8, 1], [6, 5, 1]])
    return v1, v2, M

def sequence(M : np.ndarray) -> np.ndarray:
    """
        Defines a vector given by the minimum and maximum digit of your matriculation number. Step size = 0.25.
    """
    max = np.max(M) #get the maximum value of the matrix
    min = np.min(M) #get the minimum value of the matrix
    #create a vector from min to max with each next value being 0.25 greater than the previous one
    result = np.arange(min, max + 0.25, 0.25) #exclusive max! -> +0.25
    return result

def matrix(M : np.ndarray) -> np.ndarray:
    """
        Defines the 15x9 block matrix as described in the task description.
        !!! missing: middle part
    """
    # M = black fields of chess board (3x3 matrix with matriculation number in format DBC, BGA, EHF)
    # white fields of chess board is 3x3 matrix with 0s
    white_field = np.zeros((3, 3), dtype=int)

    # create the 15x9 block matrix
    r = np.zeros((15, 9), dtype=int)

    # corners of the 15x9 block matrix are filled with black fields
    # stack the black fields with the white fields to create the top and bottom row
    # so combine first row of M with first row of white_field and first row of M as new first row of r
    top_and_bottom_row = np.hstack((M, white_field, M))
    # fill the 15x9 block matrix with the top and bottom row
    # therefore replace the first 3 and the last 3 ROWS
    r[0:3] = top_and_bottom_row
    r[12:15] = top_and_bottom_row
    
    return r


def dot_product(v1:np.ndarray, v2:np.ndarray) -> float:
    """
        Dot product of v1 and v2.
        => sum of the element-wise product of v1 and v2
        => Skalarprodukt, wird zum Beispiel verwendet, um den Winkel zwischen zwei Vektoren zu berechnen
    """
    r = 0
    for i in range(len(v1)):
        r += v1[i] * v2[i]
    return r

def cross_product(v1:np.ndarray, v2:np.ndarray) -> np.ndarray:
    """
        Cross product of v1 and v2.
        => Kreuzprodukt ist ein Vektor, der senkrecht auf der Ebene steht, die von den beiden Vektoren aufgespannt wird.
        https://en.wikipedia.org/wiki/Cross_product#Coordinate_notation -> formel
    """
    r = np.zeros(v1.shape)
    r[0] = v1[1] * v2[2] - v1[2] * v2[1]
    r[1] = v1[2] * v2[0] - v1[0] * v2[2]
    r[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return r

def vector_X_matrix(v:np.ndarray, M:np.ndarray) -> np.ndarray:
    """
        Defines the vector-matrix multiplication v*M.
        bedeutet, dass jede ZEILE der Matrix mit dem entsprechenden Element des Vektors multipliziert wird 
        und dann die Ergebnisse addiert werden
    """
    r = np.zeros((v.shape[0]), dtype=int) # same shape as v (3-Element-Vektor)
    for j in range(M.shape[1]): # iterate over columns of M
        sum_val = 0
        for i in range(v.shape[0]): # iterate over elements of a row of M
            sum_val += v[i] * M[i, j]
        r[j] = sum_val # sum of the element-wise product of v and M
    return r

def matrix_X_vector(M:np.ndarray, v:np.ndarray) -> np.ndarray:
    """
        Defines the matrix-vector multiplication M*v.
        bedeutet, dass jede SPALTE der Matrix mit dem entsprechenden Element des Vektors multipliziert wird
        und dann die Ergebnisse addiert werden
    """
    r = np.zeros((v.shape[0]), dtype=int) # same shape as v (3-Element-Vektor)
    for i in range(M.shape[0]): # iterate over rows of M
        sum_val = 0
        for j in range(M.shape[1]): # iterate over elements of a column of M
            sum_val += M[i, j] * v[j]
        r[i] = sum_val # sum of the element-wise product of M and v
    return r

def matrix_X_matrix(M1:np.ndarray, M2:np.ndarray) -> np.ndarray:
    """
        Defines the matrix multiplication M1*M2.
        https://cdn.swisscows.com/image?url=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F33%2F40%2F6f%2F33406fb252eda556c301a6ff0ee56a92.png
    """
    r = np.zeros((M2.shape[0], M1.shape[1]))
    for i in range(M2.shape[0]): # iterate over rows of M2
        for j in range(M1.shape[1]): # iterate over columns of M1
            sum_val = 0
            for k in range(M1.shape[1]): # iterate over elements of columns of M1 / elements of rows of M2
                sum_val += M1[j, k] * M2[k, i]
            r[i, j] = sum_val
    return r

def matrix_Xc_matrix(M1:np.ndarray, M2:np.ndarray) -> np.ndarray:
    """
        Defines the element-wise matrix multiplication M1*M2 (Hadamard Product).
        dh jedes Element der ersten Matrix wird mit dem entsprechenden Element 
        das die gleiche Position in der zweiten Matrix hat multipliziert
    """
    r = np.zeros(M1.shape)
    for i in range(M1.shape[0]): # iterate over rows of M1 / M2
        for j in range(M1.shape[1]): # iterate over columns of M1 / M2
            r[i, j] = M1[i, j] * M2[i, j]
    return r
