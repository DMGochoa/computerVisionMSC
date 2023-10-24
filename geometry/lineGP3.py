import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from geometry.planeGP3 import PlaneGP3
from geometry.pointGP3 import PointGP3
from geometry.geometryP2 import GeometryP2


class LineGP3(GeometryP2):
    """
    Represents a line in the projective space P3 using Plucker coordinates.

    Attributes:
        pointA (PointGP3): One of the two points that define the line.
        pointB (PointGP3): The other point that defines the line.
        W (numpy.ndarray): Span representation of the line.
        Wdual (numpy.ndarray): Dual of the span representation.
        Lmatrix (numpy.ndarray): Plucker matrix of the line.
        LmatrixDual (numpy.ndarray): Dual of the Plucker matrix.
    """

    def __init__(self, points=[], name=''):
        """
        Initializes the LineGP3 object with two points and a name.

        Args:
            points (list): List of two points that define the line.
            name (str): Name of the line.
        """
        super().__init__(points, name)
        for point in points:
            self.__verifyHomogeneousCoordinates(point)
        self.pointA = PointGP3(points[0], 'A')
        self.pointB = PointGP3(points[1], 'B')
        self.__spanRepresentationAndDual()

    def __spanRepresentationAndDual(self):
        """Obtain the span representation and dual of the line."""
        self.W = np.stack([self.pointA.vector, self.pointB.vector])
        self.Wdual = [
            PlaneGP3(points=[self.W[0], self.W[1], [1, 0, 0, 0]], name='P').mayusPi.tolist(),
            PlaneGP3(points=[self.W[0], self.W[1], [0, 1, 1, 0]], name='Q').mayusPi.tolist()
        ]
        self.__pluckerMatriz()

    def __pluckerMatriz(self):
        """Obtain the Plucker matrix of the line."""
        A = self.pointA.vector.reshape(4, 1)
        B = self.pointB.vector.reshape(4, 1)
        self.Lmatrix = A @ B.T - B @ A.T
        self.LmatrixDual = np.array([
            [0, self.Lmatrix[2, 3], -self.Lmatrix[1, 3], self.Lmatrix[1, 2]],
            [-self.Lmatrix[2, 3], 0, self.Lmatrix[0, 3], -self.Lmatrix[0, 2]],
            [self.Lmatrix[1, 3], -self.Lmatrix[0, 3], 0, self.Lmatrix[0, 1]],
            [-self.Lmatrix[1, 2], self.Lmatrix[0, 2], -self.Lmatrix[0, 1], 0]
        ])

    def estimateCoplanar(self, line):
        """
        Estimates if the given line is coplanar with the current line.

        Args:
            line (LineGP3): The line to be checked for coplanar.

        Returns:
            float: A scalar representing the coplanarity. A value close to zero indicates coplanarity.
        """
        coplanar = self.Lmatrix[0,1] * line.Lmatrix[0, 2] + \
        line.Lmatrix[0,1] * self.Lmatrix[2, 3] + \
        self.Lmatrix[0,2] * line.Lmatrix[3, 1] + \
        line.Lmatrix[0,2] * self.Lmatrix[3, 1] + \
        self.Lmatrix[0,3] * line.Lmatrix[1, 2] + \
        line.Lmatrix[0,3] * self.Lmatrix[1, 2]
        return coplanar

    def __verifyHomogeneousCoordinates(self, array):
        """
        Checks if the array has homogeneous coordinates.

        Args:
            array (list or numpy.ndarray): The array to be checked.

        Raises:
            ValueError: If the array does not have four elements.
        """
        if len(array) != 4:
            raise ValueError(f"The point must have four elements. {array}")

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
        if V[-1, -1] == 0:
            result = np.array([V[:, -1]])
        else:
            result = np.array([V[:, -1]])/V[-1, -1]
        return result.round(round)[0]

    def intersectionPlane(self, plane):
        """
        Calculates the intersection of the line with a plane.
        
        Args:
            plane (PlaneGP3): The plane to intersect with.
        
        Returns:
            numpy.ndarray: The intersection point.
        """
        value = np.concatenate((self.Wdual, plane.mayusPi.reshape(1,4)), axis=0)
        return self.__dataFitting(value)
        

    def plot(self, ax, x_lim=[-10, 10], y_lim=[-10, 10], z_lim=[-10, 10]):
        """
        Plots the line in a 3D graph.
        
        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The axis to plot the line in.
            x_lim (list, optional): The x limits of the plot. Defaults to [-10, 10].
            y_lim (list, optional): The y limits of the plot. Defaults to [-10, 10].
            z_lim (list, optional): The z limits of the plot. Defaults to [-10, 10].
        """
        direction_vector = self.pointB.vector - self.pointA.vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        if np.isclose(direction_vector[1], 0) and np.isclose(direction_vector[2], 0):
            valid_intersections = [np.array([x_lim[0], 0, 0, 1]),
                                   np.array([x_lim[1], 0, 0, 1])]
        elif np.isclose(direction_vector[0], 0) and np.isclose(direction_vector[2], 0):
            valid_intersections = [np.array([0, y_lim[0], 0, 1]),
                                   np.array([0, y_lim[1], 0, 1])]
        elif np.isclose(direction_vector[0], 0) and np.isclose(direction_vector[1], 0):
            valid_intersections = [np.array([0, 0, z_lim[0], 1]),
                                   np.array([0, 0, z_lim[1], 1])]
        else:
            intersections = []
            # Planes X
            for x in x_lim:
                planeX = PlaneGP3(points=[[x, 0, 0, 1],
                                          [x, 1, 0, 1],
                                          [x, 0, 1, 1]], name='X')
                intersection = self.intersectionPlane(planeX)
                if intersection is not None:
                    intersections.append(intersection)
            # Planes Y
            for y in y_lim:
                planeY = PlaneGP3(points=[[0, y, 0, 1],
                                          [1, y, 0, 1],
                                          [0, y, 1, 1]], name='Y')
                intersection = self.intersectionPlane(planeY)
                if intersection is not None:
                    intersections.append(intersection)
            # Planes Z
            for z in z_lim:
                planeZ = PlaneGP3(points=[[0, 0, z, 1], [1, 0, z, 1], [0, 1, z, 1]], name='Z')
                intersection = self.intersectionPlane(planeZ)
                if intersection is not None:
                    intersections.append(intersection)
            # Filter intersections that are within the plotting boundaries and are not infinite
            valid_intersections = [
                point for point in intersections
                if x_lim[0] <= point[0] <= x_lim[1]
                and y_lim[0] <= point[1] <= y_lim[1]
                and z_lim[0] <= point[2] <= z_lim[1]
                and not np.any(np.isinf(point))
            ]
            if not valid_intersections:
                return

        # Use the first and last points in the sorted list as the endpoints for plotting
        start_point = valid_intersections[0]
        end_point = valid_intersections[-1]
        # Plot the line from start_point to end_point
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]])


if __name__ == '__main__':
    from geometry.pointGP3 import PointGP3
    pointA = PointGP3([1, 0, 0, 1], 'A')
    pointB = PointGP3([2, 0, 0, 1], 'B')
    lineL = LineGP3([[1, 0, 0, 1], [2, 0, 0, 1]], 'L')
    lineM = LineGP3([[-2, -1, 1, 1], [3, 3, 0, 1]], 'M')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pointA.plot(ax)
    pointB.plot(ax)
    lineL.plot(ax)
    lineM.plot(ax)
    plt.show()
