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
            self. points = self.__get3Points()
        elif points:
            self.points = points
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

    def __get3Points(self):
        xy = np.array([[0, 0],
                      [0, 1],
                      [1, 0]])
        if self.mayusPi[2] != 0:
            z = -(self.mayusPi[0]*xy[:, 0] + self.mayusPi[1]*xy[:, 1] + self.mayusPi[3]) / self.mayusPi[2]
        else:
            z = np.zeros((3, 1))
        return np.concatenate((xy, z.reshape(3, 1), np.ones((3,1))), axis=1).tolist()

    def __verifyHomogeneousCoordinates(self, array):
        """Checks if the array has homogeneous coordinates."""
        if len(array) != 4:
            raise ValueError(f"The point must have four elements. {array}")

    def equation(self):
        """Returns the equation of the plane in implicit form."""
        return f"{self.mayusPi[0]}x + {self.mayusPi[1]}y + {self.mayusPi[2]}z + {self.mayusPi[3]}w = 0"

    def plot(self, ax, x_range=(-10, 10), y_range=(-10, 10), alpha=0.5, rstride=100, cstride=100, color='b'):
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
        elif self.mayusPi[2] == 0:
            Z = np.zeros(X.shape)
        else:
            Z = (-self.mayusPi[0]*X - self.mayusPi[1]*Y - self.mayusPi[3]) / self.mayusPi[2]

        ax.plot_surface(X, Y, Z, alpha=alpha, rstride=rstride, cstride=cstride, label=self.element_name, color=color)

    def plotIntersectionPlaneQuadric(self, ax,
                                     quadric,
                                     alpha_lim=(-5,5),
                                     beta_lim=(-5,5), tetha_lim=(-5,5), resolution=700, color='black'):
        """Calculates the intersection of the plane with a quadric."""

        alpha = np.linspace(alpha_lim[0], alpha_lim[1], resolution)
        beta = np.linspace(beta_lim[0], beta_lim[1], resolution)
        tetha = np.linspace(tetha_lim[0], tetha_lim[1], resolution)
        alpha, beta, tetha = np.meshgrid(alpha, beta, tetha)

        vec = np.array([alpha.ravel(), beta.ravel(), tetha.ravel()])
        # Suponiendo que tienes definido M y matrixQ
        M = np.array(self.points).T
        result_matrix = M.T @ quadric.matrixQ @ M

        # Calcular solo los valores de la diagonal principal de forma vectorizada
        diagonal_values = np.sum(vec * (result_matrix @ vec), axis=0)
        vec_result = vec[:,abs(diagonal_values) < 4e-4]
        vec_result_p3 = M @ vec_result
        P1 = vec_result_p3.T / vec_result_p3.T[:, 3].reshape(-1, 1)
        ax.scatter(P1[:,0], P1[:,1], P1[:,2], color=color)

    def pointInPlane(self, point):
        """Checks if the point is in the plane."""
        return abs(self.mayusPi @ point.vector) < 1e-4

    def lineInPlane(self, line):
        """Checks if the line is in the plane."""
        return abs(line.W @ self.mayusPi) < 1e-4

# Example usage:
if __name__ == '__main__':
    from geometry.quadricGP3 import QuadricGP3

    points = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
    plane = PlaneGP3(points=points, name="Plane1")
    quadric = QuadricGP3(coeffs=[1, 1, 1, -16, 0, 0, 0, 0, 0, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plane.plot(ax=ax, x_range=(-4, 4), y_range=(-4, 4))
    quadric.plot(ax, xlim=(-5, 5), ylim=(-5, 5))
    plane.plotIntersectionPlaneQuadric(ax, quadric)
    plt.show()