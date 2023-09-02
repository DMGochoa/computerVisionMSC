import numpy as np
from abc import ABC, abstractmethod

class GeometryP2(ABC):

    def __init__(self, array:list, nombre_elemento:str) -> None:
        super().__init__()
        self.vector = np.array(array)
        self.nombre_elemento = nombre_elemento

    @abstractmethod
    def _productoVectorial(self, array:np.array) -> list:
        pass

    @abstractmethod
    def plot(self):
        pass

    
