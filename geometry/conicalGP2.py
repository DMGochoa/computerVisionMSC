"""This module implements the ConicalGP2 class for representing conic sections in the projective plane P2.

The ConicalGP2 class extends the GeometryP2 abstract class and adds functionalities specific for handling conic sections.
These include methods for computing tangent lines to a conic at a given point, calculating points on the conic from a tangent line,
and checking if a given point is on the conic or if a given line is tangent to the conic.


Diego A Moreno G
Msc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
from geometry.geometryP2 import GeometryP2
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np


class ConicalGP2(GeometryP2):
    """The ConicalGP2 class represents a conic section in the projective plane P2.

    Attributes:
        A, B, C, D, E, F (float): Coefficients of the conic equation.
        fconic (function): Lambda function representing the conic equation.
        conic (Sympy expression): Sympy expression representing the conic equation.
        array (np.array): 3x3 array representing the conic.
    """
    
    def __init__(self, C=None, name=None, puntos=None, round=1) -> None:
        """Initializes a ConicalGP2 object either from given coefficients or points.

        Args:
            C (np.array or list): 3x3 array representing the conic. Default is None.
            name (str): The name of the conic. Default is None.
            puntos (list): List of points used to fit the conic. Default is None.
            round (int): Decimal places to round the conic coefficients. Default is 1.
        """
        # Define the symbolic variables x, y
        x, y = sp.symbols('x y')
        
        # Initialize conic from coefficients
        if puntos is None:
            super().__init__(C, name)
            self.__verifyShapeMatrix()
            self.__verifySimetricMatrix()
            self.__verifyMatrixRank()
            self.__verifyDeterminant()
            self.A = self.array[0, 0]
            self.B = self.array[1, 1]
            self.C = self.array[2, 2]
            self.D = self.array[0, 1]
            self.E = self.array[0, 2]
            self.F = self.array[1, 2]
            self.fconic = lambda x, y: self.A*x**2 + self.B*y**2 + self.C + 2*self.D*x*y + \
                2*self.E*x + 2*self.F*y
            self.conic = self.A*x**2 + self.B*y**2 + self.C + 2*self.D*x*y + \
                2*self.E*x + 2*self.F*y
        else:
            self.__verifyMinPoint(puntos)
            A_matrix, mu, sigma = self.__normAMatrix(puntos)
            normParameters = self.__dataFitting(A_matrix, round)
            self.A, self.B, self.C, self.D, self.E, self.F = self.__denormalizeParam(
                normParameters, mu, sigma)
            self.fconic = lambda x, y: self.A*x**2 + self.B*y**2 + self.C + self.D*x*y + \
                self.E*x + self.F*y
            self.conic = self.A*x**2 + self.B*y**2 + self.C + 2*self.D*x*y + \
                2*self.E*x + 2*self.F*y
            self.array = np.array([[self.A, self.D/2, self.E/2],
                                   [self.D/2, self.B, self.F/2],
                                   [self.E/2, self.F/2, self.C]])

    @staticmethod
    def _crossProduct(arrayLeft, arrayRight):
        """Computes the cross product of two arrays.
        
        Args:
            arrayLeft (numpy.ndarray): The first array.
            arrayRight (numpy.ndarray): The second array.
            
        Returns:
            numpy.ndarray: The cross product of the two arrays.
        """
        return np.cross(arrayLeft, arrayRight)

    def calculateTanLine(self, point, name):
        """Calculates the tangent line to the conic at a given point.

        Args:
            point (PointGP2): The point on the conic where the tangent line is calculated.
            name (str): The name to give to the resulting LineGP2 object.

        Returns:
            LineGP2: The tangent line to the conic at the given point.
        """
        from geometry.lineGP2 import LineGP2
        return LineGP2(self.array @ point.vector.T, name)

    def calculatePointFromTanLine(self, line, name):
        """Calculates the point on the conic that corresponds to a given tangent line.

        Args:
            line (LineGP2): The tangent line.
            name (str): The name to give to the resulting PointGP2 object.

        Returns:
            PointGP2: The point on the conic that corresponds to the given tangent line.
        """
        from geometry.pointGP2 import PointGP2
        return PointGP2(self.__invertMatrix() @ line.vector, name)

    def isTan(self, point=None, line=None, tolerance=1e-3):
        """Checks if a given point is on the conic or if a given line is tangent to the conic.

        Args:
            point (PointGP2, optional): The point to check. Defaults to None.
            line (LineGP2, optional): The line to check. Defaults to None.
            tolerance (float, optional): The tolerance value for checking. Defaults to 1e-3.

        Returns:
            bool: True if the point is on the conic or the line is tangent, otherwise False.

        Raises:
            ValueError: If neither or both of point and line are provided.
        """
        if point is None and line is None:
            raise ValueError('You must provide a point or a line')
        if point is not None and line is not None:
            raise ValueError('You must provide a point or a line, not both')
        if point is not None:
            lineR = self.calculateTanLine(point, 'R')
            pointP = self.calculatePointFromTanLine(lineR, 'P')
            result = abs(pointP.vector.T @ self.array @
                         pointP.vector - 0) < tolerance
        else:
            pointP = self.calculatePointFromTanLine(line, 'P')
            result = abs(pointP.vector.T @ self.array @
                         pointP.vector - 0) < tolerance
        return result

    def plot(self, x=np.linspace(-10, 10, 400), y=np.linspace(-10, 10, 400)):
        """Plots the conic on a 2D graph.

        Args:
            x (np.array): An array of x-values. Defaults to np.linspace(-10, 10, 400).
            y (np.array): An array of y-values. Defaults to np.linspace(-10, 10, 400).
        """
        x, y = np.meshgrid(x, y)
        Z = self.fconic(x, y)
        plt.contour(x, y, Z, [0])

    def __verifyShapeMatrix(self, condition=(3, 3)):
        """Verifies that the shape of the coefficient matrix is 3x3.
        
        Args:
            condition (tuple): Expected shape of the array. Defaults to (3, 3).

        Raises:
            ValueError: If the array shape does not match the condition.
        """
        if self.array.shape != condition:
            raise ValueError('The array must be 3x3')
        
    def __verifySimetricMatrix(self):
        """Verifies that the coefficient matrix is symmetric.

        Raises:
            ValueError: If the array is not symmetric.
        """
        if not np.array_equal(self.array, self.array.T):
            raise ValueError('The array must be symmetric')

    def __verifyMatrixRank(self, condition=None):
        """Verifies that the coefficient matrix has full rank.
        
        Args:
            condition (int, optional): The expected rank of the matrix. Defaults to the number of columns if None.

        Raises:
            ValueError: If the array does not have full rank.
        """
        if condition is None:
            condition = self.array.shape[1]
        if np.linalg.matrix_rank(self.array) != condition:
            raise ValueError('The array must be full rank')

    def __verifyDeterminant(self, condition=0):
        """Verifies that the determinant of the coefficient matrix is not zero.
        
        Args:
            condition (float): The value that the determinant should not equal. Defaults to 0.

        Raises:
            ValueError: If the determinant equals the condition.
        """
        if np.linalg.det(self.array) == condition:
            raise ValueError(f'The array has a determinant equal to {condition}')

    def __verifyMinPoint(self, points, condition=5):
        """Verifies that enough points are provided for fitting the conic.
        
        Args:
            points (list): The list of points.
            condition (int): The minimum number of points required. Defaults to 5.

        Raises:
            ValueError: If not enough points are provided.
        """
        if len(points) < condition:
            raise ValueError(f'The array needs at least {condition} points')

    def __invertMatrix(self, array=None):
        """Computes the inverse of a given matrix using Singular Value Decomposition (SVD).
        
        Args:
            array (numpy.ndarray, optional): The matrix to be inverted. Defaults to the instance's array.

        Returns:
            numpy.ndarray: The inverted matrix.
        """
        if array is None:
            array = self.array
        U, D, V = np.linalg.svd(array)
        D = np.diag(D)
        D = np.linalg.inv(D)
        return V.T @ D @ U.T

    def __normAMatrix(self, points):
        """Normalizes the given points and constructs the A matrix for fitting.
        
        Args:
            points (list): The list of points in the form [x, y].

        Returns:
            numpy.ndarray: The normalized A matrix for conic fitting.
            numpy.ndarray: The mean of the given points.
            numpy.ndarray: The standard deviation of the given points.
        """
        pointsXY = np.array([[punto[0], punto[1]] for punto in points])
        mu = np.mean(pointsXY, axis=0)
        sigma = np.std(pointsXY, axis=0)
        normPoints = (pointsXY - mu) / sigma
        Amatrix = np.concatenate(((normPoints[:, 0]**2).reshape(-1, 1),
                                  (normPoints[:, 1]**2).reshape(-1, 1),
                                  (np.ones(len(points))).reshape(-1, 1),
                                  (normPoints[:, 0]*normPoints[:, 1]).reshape(-1, 1),
                                  (normPoints[:, 0]).reshape(-1, 1),
                                  (normPoints[:, 1]).reshape(-1, 1)), axis=1)
        return Amatrix, mu, sigma

    def __denormalizeParam(self, parameters, mu, sigma):
        """Denormalizes the fitted parameters to obtain the coefficients of the conic equation.

        Args:
            parameters (numpy.ndarray): The normalized parameters.
            mu (numpy.ndarray): The mean of the given points.
            sigma (numpy.ndarray): The standard deviation of the given points.

        Returns:
            tuple: The denormalized parameters (a, b, c, d, e, f).
        """
        a = parameters[0] / sigma[0]**2
        b = parameters[1] / sigma[1]**2
        c = parameters[0] * mu[0]**2 / sigma[0]**2 + \
            parameters[1] * mu[1]**2 / sigma[1]**2 + \
            parameters[2] + \
            parameters[3] * mu[0] * mu[1] / (sigma[0] * sigma[1]) - \
            parameters[4] * mu[0] / sigma[0] - \
            parameters[5] * mu[1] / sigma[1]
        d = parameters[3] / (sigma[0] * sigma[1])
        e = -2 * parameters[0] * mu[0] / sigma[0]**2 - \
            parameters[3] * mu[1] / (sigma[0] * sigma[1]) + \
            parameters[4] / sigma[0]
        f = -2 * parameters[1] * mu[1] / sigma[1]**2 - \
            parameters[3] * mu[0] / (sigma[0] * sigma[1]) + \
            parameters[5] / sigma[1]
        return a, b, c, d, e, f

    def __dataFitting(self, A, round=3):
        """Performs Singular Value Decomposition (SVD) to fit the conic to the given points.

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
