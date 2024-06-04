# Copyright TU Wien (2022) - EVC: Task5
# Institute of Computer Graphics and Algorithms.

from typing import List
import numpy as np

class ClippingPlane:

    def __init__(self, plane : np.ndarray):
        """ plane     ... plane stored in Hessian normal form as a 1x4 vector"""
        self.plane = plane
    
    def inside(self, pos : np.ndarray) -> bool:
        """Checks if a given point lies behind the plane (opposite direction
        of normal vector). Points lying on the plane are considered to be
        inside.
        position  ... homogeneous position with 4 components
        return res... logical value which indicates if the point is
                      inside or not """

        
        # It does this by taking the dot product of the position and the plane's normal vector.
        # If the dot product is less than or equal to 0, the position is inside (or on) the plane.
        # If the dot product is greater than 0, the position is outside the plane.
        # The plane is defined such that the normal vector points towards the inside half-space.
        return np.dot(pos, self.plane) <= 0
    

    def intersect(self, pos1 : np.ndarray, pos2 : np.ndarray) -> float:
        """ Intersects the plane with a line between pos1 and pos2.
        pos1      ... homogeneous position with 4 components
        pos2      ... homogeneous position with 4 components
        return t  ... normalized intersection value t in [0, 1]"""

        # Calculate the dot product of the positions and the plane's normal vector
        la = np.dot(pos1, self.plane)
        lb = np.dot(pos2, self.plane)

        # Calculate the intersection point 't' along the line defined by pos1 and pos2
        # 't' is the ratio of the distance from pos1 to the intersection point and the distance from pos1 to pos2
        t = la / (la - lb)

        # Numerical stability
        # If pos1 is on the outside of the plane, subtract a small value from 't' to ensure it's not exactly on the plane
        if np.dot(pos1, self.plane) < 0:
            t = max(0, t - 1e-6)
        # If pos1 is on the inside of the plane, add a small value to 't' to ensure it's not exactly on the plane
        else:
            t = min(1, t + 1e-6)

        # Return the intersection point 't'
        return t
    
    @staticmethod
    def get_clipping_planes() -> List:
        """creates and returns a list of the six Clipping planes defined in the task description."""

        # Initialize an empty list to store the clipping planes
        planes = []

        # Add the front clipping plane (normal vector points towards the viewer)
        planes.append(ClippingPlane(np.array([0, 0, 1, 1])))    # Front side

        # Add the back clipping plane (normal vector points away from the viewer)
        planes.append(ClippingPlane(np.array([0, 0, -1, 1])))   # Back side

        # Add the bottom clipping plane (normal vector points upwards)
        planes.append(ClippingPlane(np.array([0, -1, 0, 1])))   # Bottom side

        # Add the top clipping plane (normal vector points downwards)
        planes.append(ClippingPlane(np.array([0, 1, 0, 1])))    # Top side

        # Add the left clipping plane (normal vector points to the right)
        planes.append(ClippingPlane(np.array([-1, 0, 0, 1])))   # Left side

        # Add the right clipping plane (normal vector points to the left)
        planes.append(ClippingPlane(np.array([1, 0, 0, 1])))    # Right side

        # Return the list of clipping planes
        return planes

