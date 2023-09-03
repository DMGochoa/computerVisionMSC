import numpy as np
from abc import ABC, abstractmethod


class GeometryP2(ABC):

    def __init__(self, array: list, element_name: str) -> None:
        super().__init__()
        self.array = np.array(array)
        self.element_name = element_name

    @abstractmethod
    def plot(self):
        pass
