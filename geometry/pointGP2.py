import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np
from geometry.geometryP2 import GeometryP2

class PointGP2(GeometryP2):
    
    def __init__(self, array, nombre):
        super().__init__(array, nombre)

    def _productoVectorial(self, array):
        return np.cross(self.vector.T, array).tolist()

    def plot(self):
        plt.scatter(self.vector[0],
                    self.vector[1],
                    marker='o',
                    label=self.nombre_elemento)
        plt.text(self.vector[0],
                 self.vector[1],
                 self.nombre_elemento,
                 fontsize=16,
                 ha='right')

    def calcularLinea(self, punto, nombre):
        from geometry.lineGP2 import LineGP2  # Importación local
        return LineGP2(self._productoVectorial(punto.vector), nombre)

