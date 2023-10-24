"""This module implements the PointGP3 class to represent a point in the projective space P3.

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

class PointGP3(GeometryP2):
    """The PointGP3 class represents a point in the projective space P3.

    Attributes:
        vector (np.array): The homogeneous coordinates of the point.
        element_name (str): The name of the point.
    """

    def __init__(self, array=[], name='', planes=None):
        """Initializes the PointGP3 object with homogeneous coordinates and a name.

        Args:
            array (list or np.array): List or array of four elements that represent the point in the projective space P3.
                                      Should be in homogeneous coordinates [x, y, z, 1].
            name (str): Name of the point.
            planes (list of PlaneGP3 or list od lists, optional): List of three planes that define the point. Defaults to None.
        """
        if planes is None:
            super().__init__(array, name)
            self.__verifyHomogeneousCoordinates(array)
        else:
            if type(planes[0]) is not list:
               A = np.array([planes[0].mayusPi,
                             planes[1].mayusPi,
                             planes[2].mayusPi])
            else:
                A = np.array(planes) 
                estimate = self.__dataFitting(A).tolist()
                super().__init__(estimate, name)

        self.vector = self.__homogeneousCoordinates(self.array)


    def __homogeneousCoordinates(self, array):
        """Returns the homogeneous coordinates of the point."""
        return array / array[-1]

    def __dataFitting(self, A, round=3):
        """
        Performs Singular Value Decomposition (SVD) to fit the conic to the given points.

        Args:
            A (numpy.ndarray): The A matrix constructed from the points.
            round (int, optional): The number of decimal places to round the fitted parameters to. Defaults to 3.

        Returns:
            numpy.ndarray: The fitted parameters.
        """
        _, _, V = np.linalg.svd(A)
        V = V.T
        result = np.array([V[:, -1]])/V[-1, -1]
        return result.round(round)[0]

    def __verifyHomogeneousCoordinates(self, array):
        """Checks if the array has homogeneous coordinates."""
        if len(array) != 4:
            raise ValueError("The array must have four elements.")

    def _crossProduct(self, array):
        """Calculates the cross product of the point's vector with another vector.

        Args:
            array (np.array): The vector to cross with the point's vector.

        Returns:
            list: The resulting vector from the cross product.
        """
        resultingVector = np.cross(self.vector.T, array)
        return resultingVector.tolist()

    def plot(self, ax):
        """Plots the point in a 3D graph."""
        ax.scatter(self.vector[0], self.vector[1], self.vector[2], marker='o', label=self.element_name)
        ax.text(self.vector[0], self.vector[1], self.vector[2], self.element_name, fontsize=16)


if __name__ == '__main__':
    point1 = PointGP3([1, 2, 3, 1], 'P1')
    point2 = PointGP3([4, 5, 6, 1], 'P2')
    point3 = PointGP3([7, 8, 9, 1], 'P3')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point1.plot(ax)
    point2.plot(ax)
    point3.plot(ax)
    plt.show()
    
    P1 = PointGP3(name='P1', planes=[[1, 0, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 0]])
    print(P1.vector)
