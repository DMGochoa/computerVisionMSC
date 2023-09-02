"""This module implements the LineGP2 class to represent a line in the projective plane P2.

Diego A Moreno G
Msc Student in Elctrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from geometry.geometryP2 import GeometryP2

class LineGP2(GeometryP2):
    
    def __init__(self, array, nombre):
        super().__init__(array, nombre)
        self.vector = self.array

    def _crossProduct(self, array):
        resultingVector =np.cross(self.vector.T, array)
        resultingVector = resultingVector / resultingVector[-1]
        return resultingVector.tolist()

    def plot(self, x_values = np.linspace(-10, 10, 100)):
        # Implementación del método plot para LineGP2
        if self.vector[1] != 0:
            y_values = (-self.vector[0] * x_values - self.vector[2])/ self.vector[1]
        else:
            y_values = np.linspace(-10, 10, 100)
            x_values = np.linspace(-self.vector[2]/self.vector[0], -self.vector[2]/self.vector[0], 100)
        plt.plot(x_values, y_values, label=self.nombre_elemento)
        plt.text(x_values[-1],
                 y_values[-1],
                 self.nombre_elemento,
                 fontsize=16,
                 ha='right')

    def calcularPunto(self, linea, nombre):
        from geometry.pointGP2 import PointGP2  # Importación local
        return PointGP2(self._crossProduct(linea.vector), nombre)
