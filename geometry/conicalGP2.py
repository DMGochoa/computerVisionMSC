import sys
sys.path.append("./")
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from geometry.geometryP2 import GeometryP2


class ConicalGP2(GeometryP2):

    def __init__(self, C=None, nombre=None, puntos=None) -> None:
        x, y = sp.symbols('x y')
        if puntos is None:
            super().__init__(C, nombre)
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
            self.__verifyMinPoint()
            A_matrix = np.array([[punto[0]**2,
                                  punto[1]**2,
                                  1,
                                  punto[0]*punto[1],
                                  punto[0],
                                  punto[1]] for punto in puntos])
            parameters = self.__dataFitting(A_matrix)
            self.A = parameters[0]
            self.B = parameters[1]
            self.C = parameters[2]
            self.D = parameters[3]
            self.E = parameters[4]
            self.F = parameters[5]
            self.fconic = lambda x, y: self.A*x**2 + self.B*y**2 + self.C + 2*self.D*x*y + \
                        2*self.E*x + 2*self.F*y
            self.conic = self.A*x**2 + self.B*y**2 + self.C + 2*self.D*x*y + \
                        2*self.E*x + 2*self.F*y
            self.array = np.array([[self.A, self.D/2, self.E/2],
                                   [self.D/2, self.B, self.F/2],
                                   [self.E/2, self.F/2, self.C]])

    @staticmethod
    def _crossProduct(arrayLeft, arrayRight):
        return np.cross(arrayLeft, arrayRight)

    def calculateTanLine(self, point, name):
        from geometry.lineGP2 import LineGP2
        return LineGP2(self.array @ point.vector.T, name)

    def calculatePoint(self, line, name):
        from geometry.pointGP2 import PointGP2
        return PointGP2(self.__invertMatrix() @ line.vector, name)

    def plot(self, x = np.linspace(-10, 10, 400), y = np.linspace(-10, 10, 400)):
        x, y = np.meshgrid(x, y)
        Z = self.fconic(x, y)
        plt.contour(x, y, Z, [0])

    def __verifyShapeMatrix(self, condition=(3, 3)):
        if self.array.shape != condition:
            raise ValueError('The array must be 3x3')

    def __verifySimetricMatrix(self):
        if not np.array_equal(self.array, self.array.T):
            raise ValueError('The array must be symmetric')

    def __verifyMatrixRank(self, condition=None):
        if condition is None:
            condition = self.array.shape[1]
        if np.linalg.matrix_rank(self.array) != condition:
            raise ValueError('The array must be full rank')

    def __verifyDeterminant(self, condition=0):
        if np.linalg.det(self.array) == condition:
            raise ValueError(f'The array have determinant iqual to {condition}')

    def __verifyMinPoint(self, condition=5):
        if len(puntos) < condition:
            raise ValueError(f'The array needs at least {condition} points')

    def __invertMatrix(self, array=None):
        if array is None:
            array = self.array
        U, D, V = np.linalg.svd(array)
        D = np.diag(D)
        D = np.linalg.inv(D)
        return V.T @ D @ U.T

    def __dataFitting(self, A):
        _,_,V = np.linalg.svd(A)
        V = V.T
        return (np.array([V[:,-1]]).T)/V[-1,-1]

if __name__ == '__main__':
    from geometry.pointGP2 import PointGP2
    from geometry.lineGP2 import LineGP2
    array = [[1, 0, 0], [0, 1, 0], [0, 0, -25]]
    conic = ConicalGP2(array, 'C')
    # Verificar que la línea es tangente a la cónica
    pointP = PointGP2(array=[4, 3, 1], nombre='P')
    lineR = conic.calculateTanLine(pointP, 'R')
    # Verificar que el punto de corte de linea tangente
    lineR1 = LineGP2(array=[1, 0, -5], nombre='R1')
    pointX = conic.calculatePoint(lineR1, 'X')
    print(conic.conic)
    print(conic.nombre_elemento)
    plt.figure()
    # Conica
    conic.plot()
    # Tangente
    pointP.plot()
    lineR.plot()
    # Punto de corte
    lineR1.plot()
    pointX.plot()
    plt.grid()
    plt.show()
    
    puntos = [[0, 5], [5, 0], [0, -5], [-5, 0], [4, 3]]
    conic = ConicalGP2(nombre='C', puntos=puntos)
    plt.figure()
    conic.plot()
    plt.grid()
    plt.show()