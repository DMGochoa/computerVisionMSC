import numpy as np

class ConicalGP2():
    
    def __init__(self) -> None:
        pass
    
    def _productoPunto(self, array):
        return np.dot(self.vector.T, array).tolist()