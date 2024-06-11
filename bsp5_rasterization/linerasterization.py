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
    # These are the x, y, and depth (z) coordinates of the vertices in screen space
    x1, y1, depth1 = v1.get_screen_coordinates()
    x2, y2, depth2 = v2.get_screen_coordinates()

    # Get the color of the vertices
    # These are the RGB color values (as [0,1]) of the vertices eg (0.5, 0.5, 0.5)
    color1 = v1.get_color()
    color2 = v2.get_color()

    # Calculate the differences in x and y coordinates
    # This is used to determine the direction and length of the line
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Set the initial point of the line as (x1,y1)
    # This is the starting point of the line
    x = x1
    y = y1

    # Calculate the number of steps needed to draw the line
    # The number of steps is equal to the greatest difference in x or y coordinates
    # This ensures that the line is drawn smoothly, without any gaps
    steps = dx if dx > dy else dy

    # Calculate the increment in x and y coordinates
    # These are the amounts by which the x and y coordinates are increased in each step
    # They are calculated as the total difference in x or y coordinates divided by the number of steps
    x_inc = (x2 - x1) / float(steps)
    y_inc = (y2 - y1) / float(steps)

    # Draw the line by plotting the points
    for i in range(int(steps) + 1):
        # Interpolate the color and depth values
        # The interpolation factor t is calculated as the current step divided by the total number of steps
        # t is in the range [0, 1]
        # the nearer t is to 0, the nearer the color and depth are to the values of the first vertex
        # the nearer t is to 1, the nearer the color and depth are to the values of the second vertex
        t = i / steps
        color = MeshVertex.mix(color1, color2, t)
        depth = MeshVertex.mix(depth1, depth2, t)

        # Set the pixel at the current coordinates with the interpolated color and depth
        # The coordinates are rounded to the nearest integer values, as the framebuffer only accepts integer coordinates
        framebuffer.set_pixel(np.round(x).astype(int), np.round(y).astype(int), depth, color)

        # Update the current coordinates
        # The x and y coordinates are increased by their respective increments
        x += x_inc
        y += y_inc