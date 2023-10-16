import sys
sys.path.append("./")
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


class QuadricGP3:
    """La clase QuadricGP3 representa una cuádrica en el espacio proyectivo P^3."""

    def __init__(self, coeffs=[], points=None, round_val=1):
        """Inicializa un objeto QuadricGP3 a partir de coeficientes o puntos."""

        # Define las variables simbólicas x, y, z, w
        x, y, z = sp.symbols('x y z')

        if points is None:
            # Inicializar la cuádrica desde coeficientes
            self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H, self.I, self.J = coeffs
            self.fquadric = lambda x, y, z: self.A*x**2 + self.B*y**2 + self.C*z**2 + self.D + \
                            self.E*x*y + self.F*x*z + self.G*x + self.H*z*y + self.I*y + self.J*z
            self.quadric = self.A*x**2 + self.B*y**2 + self.C*z**2 + self.D**2 + self.E*x*y + \
                           self.F*x*z + self.G*x + self.H*y*z + self.I*y + self.J*z
            self.matrix = np.array([[self.A, self.E/2, self.F/2, self.G/2],
                                    [self.E/2, self.B, self.H/2, self.I/2],
                                    [self.F/2, self.H/2, self.C, self.J/2],
                                    [self.G/2, self.I/2, self.J/2, self.D]])
            self.__verifySymmetricMatrix()
            self.__verifyDeterminant()
        else:
            # Para inicializar desde puntos, necesitaríamos un método de ajuste.
            # Esta es una tarea avanzada y va más allá del alcance básico.
            # Sin embargo, puedes agregarlo si tienes un método en mente.
            pass

    def __verifySymmetricMatrix(self):
        """Verifica que la matriz sea simétrica."""
        if not np.array_equal(self.matrix, self.matrix.T):
            raise ValueError('La matriz debe ser simétrica.')

    def __verifyDeterminant(self, condition=0):
        """Verifica que el determinante de la matriz no sea cero."""
        if np.linalg.det(self.matrix) == condition:
            raise ValueError('El determinante de la matriz no debe ser cero.')

    def get_points_near_zero(self, xlim=(-10, 10), ylim=(-10, 10), zlim=(-10, 10), resolution=100, tolerance=1e-2):
        """Devuelve los puntos (x, y, z) donde F está cerca de cero.

        Args:
            tolerance (float): Tolerancia para considerar que un valor está cerca de cero. Default es 1e-2.

        Returns:
            numpy.ndarray: puntos (x, y, z) donde F está cerca de cero.
        """
        x = np.linspace(*xlim, resolution)
        y = np.linspace(*ylim, resolution)
        z = np.linspace(*zlim, resolution)
        
        x, y, z = np.meshgrid(x, y, z)
        F = self.fquadric(x, y, z)

        mask = np.abs(F) < tolerance

        x_near_zero = x[mask]
        y_near_zero = y[mask]
        z_near_zero = z[mask]

        return x_near_zero, y_near_zero, z_near_zero

    # def plot(self, xlim=(-10, 10), ylim=(-10, 10), zlim=(-10, 10), resolution=100, tolerance=1e-2):
    #     """Plots the surface of a quadric given a set of points.
        
    #     Args:
    #         points (numpy.ndarray): An Nx3 array of points representing the quadric surface.
    #     """
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     x, y, z = self.get_points_near_zero(xlim, ylim, zlim, resolution, tolerance)
    #     print(x)
    #     # Triangulate the points for a better surface plot
    #     ax.plot_wireframe(x, y, z, cmap='viridis', edgecolor='none', alpha=0.6)
        
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.show()

    # def get_points_near_zero(self, F, threshold=0.1, step=0.001):
    #     """Get points close to zero in the quadric function F.
        
    #     Args:
    #         F (function): The quadric function.
    #         threshold (float, optional): The range around zero to consider points. Defaults to 0.1.
    #         step (float, optional): The step size for the grid. Defaults to 0.5.
        
    #     Returns:
    #         numpy.ndarray: An array of points close to zero.
    #     """
    #     x = np.arange(-10, 10, step)
    #     y = np.arange(-10, 10, step)
    #     z = np.arange(-10, 10, step)
        
    #     close_to_zero_points = []

    #     for xi in x:
    #         for yi in y:
    #             for zi in z:
    #                 if abs(F(xi, yi, zi)) <= threshold:
    #                     close_to_zero_points.append([xi, yi, zi])
        
    #     return np.array(close_to_zero_points)

    def plot(self, xlim=(-10, 10), ylim=(-10, 10), zlim=(-10, 10), resolution=100, tolerance=1e-2):
        """Plots the surface of the quadric."""

        points = self.get_points_near_zero(xlim, ylim, zlim, resolution, tolerance)
        
        # Check if the points array is empty or not
        if len(points) == 0:
            print("No points found near zero.")
            return
        
        # Ensure that the points array is 2-dimensional
        points = np.array(points)
        if points.ndim != 2:
            print("Invalid points array dimension.")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Triangulate the points for a better surface plot
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], cmap='viridis', edgecolor='none', alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


# Ejemplo de uso:
quadric = QuadricGP3(coeffs=[1, 1, 1, -4, 0, 0, 0, 0, 0, 1])  # Representa x^2 + y^2 + z^2 + w^2 = 0
quadric.plot(xlim=(-4, 4), ylim=(-4, 4), zlim=(-4, 4), resolution=1000, tolerance=1e-2)
