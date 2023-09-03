"""This module implements the PointGP2 class to represent a point in the projective plane P2.

Diego A Moreno G
Msc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np
from geometry.geometryP2 import GeometryP2

class PointGP2(GeometryP2):
    """The PointGP2 class represents a point in the projective plane P2.

    Attributes:
        vector (np.array): The homogeneous coordinates of the point.
        element_name (str): The name of the point.
    """

    def __init__(self, array, name):
        """Initializes the PointGP2 object with homogeneous coordinates and a name.

        Args:
            array (list or np.array): List or array of three elements that represent the point in the projective plane P2.
                                      Should be in homogeneous coordinates [x, y, 1].
            name (str): Name of the point.
        """
        super().__init__(array, name)
        if self.array[-1] != 1:
            self.vector = self.array / self.array[-1]
        else:
            self.vector = self.array

    def _crossProduct(self, array):
        """Calculates the cross product of the point's vector with another vector.

        Args:
            array (np.array): The vector to cross with the point's vector.

        Returns:
            list: The resulting vector from the cross product.
        """
        resultingVector = np.cross(self.vector.T, array)
        return resultingVector.tolist()

    def plot(self):
        """Plots the point on a 2D graph."""
        plt.scatter(self.vector[0], self.vector[1], marker='o', label=self.element_name)
        plt.text(self.vector[0], self.vector[1], self.element_name, fontsize=16, ha='right')

    def calcularLinea(self, point, name):
        """Calculates the line passing through this point and another point.

        Args:
            point (PointGP2): The other point through which the line passes.
            name (str): The name of the resulting line.

        Returns:
            LineGP2: A LineGP2 object representing the line passing through the two points.
        """
        from geometry.lineGP2 import LineGP2
        return LineGP2(self._crossProduct(point.vector), name)
