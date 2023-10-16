import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from geometry.geometryP2 import GeometryP2


class PlaneGP3(GeometryP2):
    """Represents a plane in the projective space P3."""

    def __init__(self, coefficients=None, name='', points=None):
        """
        Initializes the PlaneGP3 object.

        Args:
            coefficients (list): List of four numbers [a, b, c, d] representing the plane's equation.
            name (str): Name of the plane.
            points (list of lists): List of three points that define the plane.
        """
        if coefficients:
            super().__init__(coefficients, name)
            self.mayusPi = np.array(coefficients)
        elif points:
            coefficients = self.__determinePlaneEquation(points)
            super().__init__(coefficients, name)
            self.mayusPi = np.array(coefficients)

    def __determinePlaneEquation(self, points):
        """Determines the equation of the plane given three points."""
        coefficients = []
        for point in points:
            self.__verifyHomogeneousCoordinates(point)
        self.X_points = np.array(points).T
        for row in range(self.X_points.shape[0]):
            X_aux = np.delete(self.X_points, row, 0)
            coefficients.append(((-1)**row) * np.linalg.det(X_aux))
        return coefficients

    def __verifyHomogeneousCoordinates(self, array):
        """Checks if the array has homogeneous coordinates."""
        if len(array) != 4:
            raise ValueError(f"The point must have four elements. {array}")

    def equation(self):
        """Returns the equation of the plane in implicit form."""
        return f"{self.mayusPi[0]}x + {self.mayusPi[1]}y + {self.mayusPi[2]}z + {self.mayusPi[3]}w = 0"

    def plot(self, ax, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), alpha=0.5, rstride=100, cstride=100, color='b'):
        """Plots the plane in a 3D graph."""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)

        if self.mayusPi[0] == 0 and self.mayusPi[1] == 0 and self.mayusPi[2] == 0:
            raise ValueError("The plane is degenerate.")
        elif self.mayusPi[0] == 0 and self.mayusPi[1] == 0:
            Z = np.zeros(X.shape)
        elif self.mayusPi[0] == 0 and self.mayusPi[2] == 0:
            Z = Y
            Y = np.zeros(Y.shape)
        elif self.mayusPi[1] == 0 and self.mayusPi[2] == 0:
            Z = X
            X = np.zeros(X.shape)
        else:
            Z = (-self.mayusPi[0]*X - self.mayusPi[1]*Y - self.mayusPi[3]) / self.mayusPi[2]

        ax.plot_surface(X, Y, Z, alpha=alpha, rstride=rstride, cstride=cstride, label=self.element_name, color=color)

# Example usage:
if __name__ == '__main__':
    points = [[1, 5, 2, 1], [1, 2, 1, 1], [2, 0, 1, 1]]
    plane = PlaneGP3(points=points, name="Plane1")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(plane.equation())
    plane.plot(ax)
    plt.show()