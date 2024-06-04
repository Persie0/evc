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

def draw_triangle(framebuffer: Framebuffer, v1: MeshVertex, v2: MeshVertex, v3: MeshVertex):
    """ draws a triangle defined by v1,v2,v3 to the given framebuffer"""
    x1, y1, depth1 = v1.get_screen_coordinates()
    x2, y2, depth2 = v2.get_screen_coordinates()
    x3, y3, depth3 = v3.get_screen_coordinates()

    col1 = v1.get_color()
    col2 = v2.get_color()
    col3 = v3.get_color()

    # calc triangle area * 2
    a = ((x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1))

    if not np.isclose(a, 0):
        # Swap order of clockwise triangle to make them counter-clockwise
        if a < 0:
            x2, x3 = x3, x2
            y2, y3 = y3, y2
            depth2, depth3 = depth3, depth2
            col2, col3 = col3, col2

        # Calculate edge equations
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y3 - y2
        B2 = x2 - x3
        C2 = A2 * x2 + B2 * y2

        A3 = y1 - y3
        B3 = x3 - x1
        C3 = A3 * x3 + B3 * y3

        # Calculate bounding box
        x_min = max(min(x1, x2, x3), 0)
        x_max = min(max(x1, x2, x3), framebuffer.width - 1)
        y_min = max(min(y1, y2, y3), 0)
        y_max = min(max(y1, y2, y3), framebuffer.height - 1)

        # Iterate over bounding box
        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                # Test if the point is inside the triangle
                inside1 = line_eq(A1, B1, C1, x, y) <= 0
                inside2 = line_eq(A2, B2, C2, x, y) <= 0
                inside3 = line_eq(A3, B3, C3, x, y) <= 0

                if inside1 and inside2 and inside3:
                    # Compute barycentric coordinates
                    f1 = line_eq(A1, B1, C1, x, y) / line_eq(A1, B1, C1, x1, y1)
                    f2 = line_eq(A2, B2, C2, x, y) / line_eq(A2, B2, C2, x2, y2)
                    f3 = line_eq(A3, B3, C3, x, y) / line_eq(A3, B3, C3, x3, y3)

                    alpha = f1
                    beta = f2
                    gamma = f3

                    # Interpolate color and depth
                    color = MeshVertex.barycentric_mix(col1, col2, col3, alpha, beta, gamma)
                    depth = MeshVertex.barycentric_mix(depth1, depth2, depth3, alpha, beta, gamma)

                    # Set pixel in framebuffer
                    framebuffer.set_pixel(x, y, depth, color)