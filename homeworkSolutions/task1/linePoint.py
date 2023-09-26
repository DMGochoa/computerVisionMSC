"""This module implements an example for the creation and using of points and lines
class on the P2 space.

Diego A Moreno G
Msc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
from geometry.lineGP2 import LineGP2
from geometry.pointGP2 import PointGP2

# In this section we will instaciate a line and a point
pointP = PointGP2(array=[1, 2, 1],
                  name='P')
pointQ = PointGP2(array=[4, 5, 1],
                  name='Q')
lineL = LineGP2(array=[-3, 6, -3],
                name='L')

# We create the line R that passes through P and Q
lineR = pointP.calcularLinea(pointQ, name='R')
print('The resulting line is: ',lineR.vector)
# We create the point X that is the intersection of L and R
pointX = lineL.calcularPunto(lineR, name='X')
print('The resulting point is: ',pointX.vector)

plt.figure()
pointP.plot()
pointQ.plot()
lineR.plot()
lineL.plot()
pointX.plot()
plt.grid()
plt.legend()
plt.pause(6)
