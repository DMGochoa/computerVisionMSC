import sys
sys.path.append("./")
from geometry.pointGP2 import PointGP2
#import geometryP2.pointGP2 as pgp2

pointP = PointGP2(array=[1, 2, 1],
                  nombre='P')
pointQ = PointGP2(array=[4, 5, 1],
                  nombre='Q')
    
lineR = pointP.calcularLinea(pointQ, nombre='R')
print(lineR.vector)