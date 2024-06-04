# Copyright TU Wien (2022) - EVC: Task 5
# Institute of Computer Graphics and Algorithms.

import numpy as np
from numpy.matlib import repmat

from Mesh import Mesh
from Framebuffer import Framebuffer
from MeshVertex import MeshVertex

def line_rasterization(mesh : Mesh, framebuffer : Framebuffer):
    """ iterates over all faces of mesh and draws lines between
        their vertices.
        mesh                  ... mesh object to rasterize
        framebuffer           ... framebuffer"""

    for i in range(mesh.faces.shape[0]):
        for j in range(mesh.faces[i][0]):
            i, j = np.array(i).reshape(np.asarray(i).size), np.array(j).reshape(np.asarray(j).size)

            v1 = mesh.get_face(i).get_vertex(j)
            v2 = mesh.get_face(i).get_vertex(np.remainder(j + 1, mesh.faces[i]))
            draw_line(framebuffer, v1, v2)

def draw_line(framebuffer: Framebuffer, v1: MeshVertex, v2: MeshVertex):
    """
    Draws a line between v1 and v2 into the framebuffer using the
    DDA (Digital Differential Analyzer) algorithm.
    
    Parameters:
    framebuffer: The framebuffer where the line will be drawn.
    v1: The first vertex of the line.
    v2: The second vertex of the line.
    """

    # Get the screen coordinates of the vertices
    x1, y1, depth1 = v1.get_screen_coordinates()
    x2, y2, depth2 = v2.get_screen_coordinates()

    # Calculate the differences in x and y coordinates
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # The number of steps is the maximum difference in x or y
    steps = max(dx, dy)

    # Calculate the increments in x and y for each step
    dx /= steps if steps != 0 else dx
    dy /= steps if steps != 0 else dy

    # Start from the first vertex
    x = x1
    y = y1

    # Loop over the steps
    for i in range(int(steps) + 1):
        # Calculate the interpolation coefficient
        t = i / steps

        # Interpolate the color and depth for the current step
        interpolated_color = v1.mix(v2, t).get_color()
        interpolated_depth = v1.mix(v2, t).get_depth()

        # Set the pixel at the current coordinates with the interpolated color and depth
        framebuffer.set_pixel(int(round(x)), int(round(y)), interpolated_depth, interpolated_color)

        # Move to the next coordinates
        x += dx
        y += dy
