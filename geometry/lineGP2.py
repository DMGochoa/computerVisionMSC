import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from geometry.geometryP2 import GeometryP2

class LineGP2(GeometryP2):
    
    def __init__(self, array, nombre):
        super().__init__(array, nombre)

    def _productoVectorial(self, array):
        resultingVector =np.cross(self.vector.T, array)
        resultingVector = resultingVector / resultingVector[-1]
        return resultingVector.tolist()

    def plot(self, x_values = np.linspace(-10, 10, 100)):
        # Implementación del método plot para LineGP2
        y_values = (-self.vector[0] * x_values - self.vector[2])/ self.vector[1]
        plt.plot(x_values, y_values, label=self.nombre_elemento)

    def calcularPunto(self, linea, nombre):
        from geometry.pointGP2 import PointGP2  # Importación local
        return PointGP2(self._productoVectorial(linea.vector), nombre)


if __name__ == '__main__':
    lineP = LineGP2(array=[1, 2, 3], nombre='P')
    lineQ = LineGP2(array=[4, 5, 6], nombre='Q')
    
    pointR = lineP.calcularPunto(lineQ, nombre='R')
    print(pointR.vector)
    
    plt.figure()
    lineP.plot()
    lineQ.plot()
    pointR.plot()
    plt.legend()
    
    plt.show()
