# Copyright TU Wien (2022) - EVC: Task5
# Institute of Computer Graphics and Algorithms.

from copy import copy
from typing import List

import numpy as np

from Mesh import Mesh
from MeshVertex import MeshVertex
from ClippingPlane import ClippingPlane


def clip(mesh : Mesh, planes : List[ClippingPlane]) -> Mesh:
    """ clip the mesh with the given planes."""
    clipped_mesh = copy(mesh)
    clipped_mesh.clear()

    for f in range(mesh.faces.shape[0]):
        vertices = mesh.get_face(f).get_vertex(np.arange(mesh.faces[f]))

        positions = vertices.get_position()
        colors = vertices.get_color()
        vertex_count = 3

        for plane in planes:
            vertex_count, positions, colors = clip_plane(vertex_count, positions, colors, plane)

        if vertex_count != 0:
            clipped_mesh.add_face(vertex_count, positions, colors)

    return clipped_mesh

def clip_plane(vertex_count : int, positions : np.ndarray, colors : np.ndarray, plane : ClippingPlane) -> List[np.ndarray]:
    """ clips all vertices defined in positions against the clipping
             plane clipping_plane. Clipping is done by using the Sutherland
             Hodgman algorithm.

        Input Parameter
            vertex_count          ... number of vertices of the face that is clipped
            positions             ... n x 4 matrix with positions of n vertices
                                    one row corresponds to one vertex position
            colors                ... n x 3 matrix with colors of n vertices
                                    one row corresponds to one vertex color
            plane                 ... plane to clip against

        Returns:
            vertex_count_clipped  ... number of resulting vertices after clipping;
                                    this number depends on how the plane intersects
                                    with the face and therefore is not constant
            pos_clipped           ... n x 4 matrix with positions of n clipped vertices
                                    one row corresponds to one vertex position
            col_clipped           ... n x 3 matrix with colors of n clipped vertices
                                    one row corresponds to one vertex color"""
 
    # Initialize arrays for storing the clipped positions and colors
    pos_clipped = np.zeros((vertex_count + 1, 4))
    col_clipped = np.zeros((vertex_count + 1, 3))
    vertex_count_clipped = 0

    # Loop over all edges of the polygon
    for i in range(vertex_count):
        # Get the current and next vertices and their colors
        v1 = positions[i]
        v2 = positions[(i + 1) % vertex_count]
        c1 = colors[i]
        c2 = colors[(i + 1) % vertex_count]

        # Check if the edge intersects the plane
        inside1 = plane.inside(v1)
        inside2 = plane.inside(v2)

        # If v1 is inside the plane and v2 is outside
        if inside1 and not inside2:
            # Compute the intersection point of the edge and the plane
            t = plane.intersect(v1, v2)
            # Compute the position and color at the intersection point
            pos_clipped[vertex_count_clipped] = v1 * (1 - t) + v2 * t
            col_clipped[vertex_count_clipped] = c1 * (1 - t) + c2 * t
            vertex_count_clipped += 1

            # Add the outside vertex to the clipped polygon
            pos_clipped[vertex_count_clipped] = v2
            col_clipped[vertex_count_clipped] = c2
            vertex_count_clipped += 1

        # If v2 is inside the plane and v1 is outside
        elif inside2 and not inside1:
            # Compute the intersection point of the edge and the plane
            t = plane.intersect(v2, v1)
            # Compute the position and color at the intersection point
            pos_clipped[vertex_count_clipped] = v2 * (1 - t) + v1 * t
            col_clipped[vertex_count_clipped] = c2 * (1 - t) + c1 * t
            vertex_count_clipped += 1

            # Add the outside vertex to the clipped polygon
            pos_clipped[vertex_count_clipped] = v1
            col_clipped[vertex_count_clipped] = c1
            vertex_count_clipped += 1

        # If both vertices are inside the plane
        elif inside1 and inside2:
            # Add the next vertex to the clipped polygon
            pos_clipped[vertex_count_clipped] = v2
            col_clipped[vertex_count_clipped] = c2
            vertex_count_clipped += 1

    # Return the number of vertices in the clipped polygon and the clipped positions and colors
    return vertex_count_clipped, pos_clipped[:vertex_count_clipped], col_clipped[:vertex_count_clipped]