"""This module implements the PointGP2 class to represent a point in the projective plane P2.

Diego A Moreno G
Msc Student in Elctrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np
from geometry.geometryP2 import GeometryP2

class PointGP2(GeometryP2):
    
    def __init__(self, array, nombre):
        super().__init__(array, nombre)
        if self.array[-1] != 1:
            self.vector = self.array / self.array[-1]
        else:
            self.vector = self.array

    def _crossProduct(self, array):
        resultingVector = np.cross(self.vector.T, array)
        return resultingVector.tolist()

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
        return LineGP2(self._crossProduct(punto.vector), nombre)

