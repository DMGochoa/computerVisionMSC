"""This module implements the LineGP2 class to represent a line in the projective plane P2.

Diego A Moreno G
Msc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from geometry.geometryP2 import GeometryP2

class LineGP2(GeometryP2):
    """The LineGP2 class represents a line in the projective plane P2.

    Attributes:
        vector (np.array): The homogeneous coordinates of the line.
        element_name (str): The name of the line.
    """

    def __init__(self, array, name):
        """Initializes the LineGP2 object with a vector and a name.

        Args:
            array (list or np.array): List or array of three elements that represent the line in the projective plane P2.
            name (str): Name of the line.
        """
        super().__init__(array, name)
        self.vector = self.array

    def _crossProduct(self, array):
        """Calculates the cross product of the line's vector with another vector.

        Args:
            array (np.array): The vector to cross with the line's vector.

        Returns:
            list: The resulting vector from the cross product, normalized to homogeneous coordinates.
        """
        resultingVector = np.cross(self.vector.T, array)
        resultingVector = resultingVector / resultingVector[-1]
        return resultingVector.tolist()

    def plot(self, x_values=np.linspace(-10, 10, 100)):
        """Plots the line on a 2D graph.

        Args:
            x_values (np.array): An array of x-values for which to calculate the corresponding y-values. Defaults to np.linspace(-10, 10, 100).
        """
        if self.vector[1] != 0:
            y_values = (-self.vector[0] * x_values - self.vector[2]) / self.vector[1]
        else:
            y_values = np.linspace(-10, 10, 100)
            x_values = np.linspace(-self.vector[2] / self.vector[0], -self.vector[2] / self.vector[0], 100)
        plt.plot(x_values, y_values, label=self.element_name)
        plt.text(x_values[-1], y_values[-1], self.element_name, fontsize=16, ha='right')

    def calcularPunto(self, line, name):
        """Calculates the intersection point of this line with another line.

        Args:
            line (LineGP2): The other line with which to find an intersection.
            name (str): The name of the resulting intersection point.

        Returns:
            PointGP2: A PointGP2 object representing the intersection point of the two lines.
        """
        from geometry.pointGP2 import PointGP2
        return PointGP2(self._crossProduct(line.vector), name)

