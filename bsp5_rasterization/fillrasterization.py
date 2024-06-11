# Copyright TU Wien (2022) - EVC: Task5
# Institute of Computer Graphics and Algorithms.

import numpy as np
from numpy.matlib import repmat

from MeshVertex import MeshVertex
from Framebuffer import Framebuffer
from MeshVertex import MeshVertex

def fill_rasterization(mesh : MeshVertex, framebuffer : Framebuffer):
    """ applies the fill rasterization algorithm. Draws a mesh to the Framebuffer."""

    for i in range(mesh.faces.shape[0]):
        v1 = mesh.get_face(i).get_vertex(0)
        for j in range(mesh.faces[i][0]-1):
            i, j = np.array(i).reshape(np.asarray(i).size), np.array(j).reshape(np.asarray(j).size)

            v2 = mesh.get_face(i).get_vertex(j)
            v3 = mesh.get_face(i).get_vertex(j+1)
            draw_triangle(framebuffer, v1, v2, v3)


def line_eq(A : float, B : float, C : float, x : float, y : float) -> float:
    """defines the line equation described by the provided parameters and
        returns the distance of a point (x, y) to this line.
        A    ... line equation parameter 1
        B    ... line equation parameter 2
        C    ... line equation parameter 3
        x    ... x coordinate of point to test against the line
        y    ... y coordinate of point to test against the line
        res  ... distance of the point (x, y) to the line (A, B, C)."""

    # This function calculates the value of a linear equation in two variables, x and y.
    # A, B, and C are the coefficients of the equation.
    # The equation is of the form: Ax + By + C = 0
    # This is used in the context of rasterization to determine the position of a point relative to a line.
    return A * x + B * y + C

def draw_triangle(framebuffer : Framebuffer, v1 : MeshVertex, v2 : MeshVertex, v3 : MeshVertex):
    """ draws a triangle defined by v1,v2,v3 to the given framebuffer"""
    
    # Get the screen coordinates and color of the vertices
    x1, y1, depth1 = v1.get_screen_coordinates()
    x2, y2, depth2 = v2.get_screen_coordinates()
    x3, y3, depth3 = v3.get_screen_coordinates()
    col1 = v1.get_color()
    col2 = v2.get_color()
    col3 = v3.get_color()

    # Calculate the area of the triangle
    a = ((x3-x1)*(y2-y1) - (x2-x1)*(y3-y1))

    #If the area is close to 0, the triangle are very close to being in a straight line
    if not np.isclose(a, 0):
        # If the vertices are ordered clockwise, swap the order to make them counter-clockwise
        # This is necessary because the algorithm assumes counter-clockwise vertex order
        if a < 0:
            t = x2
            x2 = x3 
            x3 = t

            t = y2
            y2 = y3
            y3 = t

            t = depth2
            depth2 = depth3
            depth3 = t

            t = col2
            col2 = col3
            col3 = t

    # Calculate the edge vectors of the triangle
    e1 = np.array([x3-x2, y3-y2])
    e2 = np.array([x1-x3, y1-y3])
    e3 = np.array([x2-x1, y2-y1])

    # Calculate the coefficients of the line equations for the edges as in task instructions
    # These are used to describe the edges of the triangle (lines between the vertices)
    A1 = -e1[1]
    B1 = e1[0]
    C1 = -A1 * x2 - B1 * y2

    A2 = -e2[1]
    B2 = e2[0]
    C2 = -A2 * x3 - B2 * y3

    A3 = -e3[1]
    B3 = e3[0]
    C3 = -A3 * x1 - B3 * y1

    # Calculate the bounding box of the triangle
    # The bounding box is the smallest rectangle that contains the triangle
    # also ignoring pixels outside the framebuffer
    x_min = max(min(x1, x2, x3), 0)
    x_max = min(max(x1, x2, x3), framebuffer.width - 1)
    y_min = max(min(y1, y2, y3), 0)
    y_max = min(max(y1, y2, y3), framebuffer.height - 1)

    # Calculate the barycentric coordinates of the vertices, precomputed part of the barycentric interpolation
    f1 = 1/line_eq(A1, B1, C1, x1, y1)
    f2 = 1/line_eq(A2, B2, C2, x2, y2)
    f3 = 1/line_eq(A3, B3, C3, x3, y3)


    # Iterate over the pixels in the bounding box
    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            # Test if the pixel is inside the triangle using the line equations
            # A pixel is inside the triangle if the equations of all edges are less than or equal to 0
            # equation of a line: Ax + By + C = 0
            inside1 = line_eq(A1, B1, C1, x, y) <= 0
            inside2 = line_eq(A2, B2, C2, x, y) <= 0
            inside3 = line_eq(A3, B3, C3, x, y) <= 0

            # if all the line equations are less than or equal to 0, the pixel is inside the triangle
            if inside1 and inside2 and inside3:
                # Compute the barycentric coordinates of the pixel
                # The barycentric coordinates represent the pixel's position within the triangle

                # calculate the barycentric coordinates of the pixel as in task instructions    
                alpha = line_eq(A1, B1, C1, x, y) * f1
                beta = line_eq(A2, B2, C2, x, y) * f2   
                gamma = line_eq(A3, B3, C3, x, y) * f3

                # Interpolate the color and depth of the pixel
                # The color and depth are weighted averages of the colors and depths of the vertices
                color = MeshVertex.barycentric_mix(col1, col2, col3, alpha, beta, gamma)
                depth = MeshVertex.barycentric_mix(depth1, depth2, depth3, alpha, beta, gamma)

                # Set the pixel in the framebuffer
                framebuffer.set_pixel(np.array([x]), np.array([y]), depth, color)
