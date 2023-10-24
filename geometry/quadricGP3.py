import sys
sys.path.append("./")
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


class QuadricGP3:
    """La clase QuadricGP3 representa una cuádrica en el espacio proyectivo P^3."""

    def __init__(self, coeffs=[], points=None, name=''):
        """Inicializa un objeto QuadricGP3 a partir de coeficientes o puntos."""

        # Define las variables simbólicas x, y, z, w
        x, y, z = sp.symbols('x y z')
        self.name = name
        if points is None:
            # Inicializar la cuádrica desde coeficientes
            self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H, self.I, self.J = coeffs
            self.fquadric = lambda x, y, z: self.A*x**2 + self.B*y**2 + self.C*z**2 + self.D + \
                self.E*x*y + self.F*x*z + self.G*x + self.H*z*y + self.I*y + self.J*z
            self.quadric = self.A*x**2 + self.B*y**2 + self.C*z**2 + self.D + self.E*x*y + \
                self.F*x*z + self.G*x + self.H*y*z + self.I*y + self.J*z
            self.matrixQ = np.array([[self.A, self.E/2, self.F/2, self.G/2],
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
        if not np.array_equal(self.matrixQ, self.matrixQ.T):
            raise ValueError('La matriz debe ser simétrica.')

    def __verifyDeterminant(self, condition=0):
        """Verifica que el determinante de la matriz no sea cero."""
        if np.linalg.det(self.matrixQ) == condition:
            raise ValueError('El determinante de la matriz no debe ser cero.')

    def __get_points(self, xlim=(-10, 10), ylim=(-10, 10), resolution=100):
        """Devuelve los puntos (x, y, z) donde F está cerca de cero."""
        x, y, z = sp.symbols('x y z')
        resultFunctionZ = sp.solve(self.quadric, z)

        X = np.linspace(xlim[0], xlim[1], resolution)
        Y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(X, Y)
        Z = None

        if len(resultFunctionZ) == 1:
            functionNumpyZ = sp.lambdify((x, y), resultFunctionZ[0], "numpy")
            Z = functionNumpyZ(X, Y)
        elif len(resultFunctionZ) == 2:
            functionNumpyZ1 = sp.lambdify((x, y), resultFunctionZ[0], "numpy")
            functionNumpyZ2 = sp.lambdify((x, y), resultFunctionZ[1], "numpy")
            Z1 = functionNumpyZ1(X, Y)
            Z2 = functionNumpyZ2(X, Y)
            Z = np.concatenate((Z1, Z2), axis=0)
            X = np.concatenate((X, X), axis=0)
            Y = np.concatenate((Y, Y), axis=0)
        return X, Y, Z

    def plot(self, ax, xlim=(-10, 10), ylim=(-10, 10), resolution=5000):
        """Plots the surface of the quadric."""
        X, Y, Z = self.__get_points(xlim, ylim, resolution)

        # Graficar la superficie combinada
        ax.plot_surface(X, Y, Z, alpha=0.6)

# Example usage:
if __name__ == '__main__':
    # Ejemplo de uso:
    # Representa x^2 + y^2 + z^2 + w^2 = 0
    quadric = QuadricGP3(coeffs=[1, 1, 1, -16, 0, 0, 0, 0, 0, 0])
    print(quadric.quadric)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    quadric.plot(ax, xlim=(-10, 10), ylim=(-10, 10))
    plt.show()
