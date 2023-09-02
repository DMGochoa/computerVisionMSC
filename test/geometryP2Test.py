import sys
sys.path.append("./")
from geometry.pointGP2 import PointGP2
from geometry.lineGP2 import LineGP2
import matplotlib.pyplot as plt

pointP = PointGP2(array=[1, 2, 1],
                  nombre='P')
pointQ = PointGP2(array=[4, 5, 1],
                  nombre='Q')
lineL = LineGP2(array=[-3, 6, -3],
                nombre='L')
    
lineR = pointP.calcularLinea(pointQ, nombre='R')
print(lineR.vector)
pointX = lineL.calcularPunto(lineR, nombre='X')

plt.figure()
pointP.plot()
pointQ.plot()
lineR.plot()
lineL.plot()
pointX.plot()
plt.grid()
plt.legend()
plt.show()