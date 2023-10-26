import sys
sys.path.append("./")
from geometry.pointGP3 import PointGP3
from geometry.planeGP3 import PlaneGP3
from geometry.lineGP3 import LineGP3
from geometry.quadricGP3 import QuadricGP3
import matplotlib.pyplot as plt
# ¿En que punto X, la linea que une los puntos A y B intersecta al plano Pi ?
pointA = PointGP3([-2, -2, 0, 1], 'A')
pointB = PointGP3([7, 7, 0, 1], 'B')

planePi = PlaneGP3(coefficients=[3, 1, 1, -10], name='Pi')

lineAB = LineGP3(points=[pointA.vector, pointB.vector], name='AB')

pointX = PointGP3(lineAB.intersectionPlane(planePi).tolist(), 'X')

print('Punto de interseccion: ', pointX.vector)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pointA.plot(ax)
pointB.plot(ax)
planePi.plot(ax, x_range=(-12, 12), y_range=(-14, 14))
lineAB.plot(ax)
pointX.plot(ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# ¿Cuál es el plano Pi que contiene una linea W y un punto X que no pertenece a W?

pointA = PointGP3([-2, -2, 0, 1], 'A')
pointB = PointGP3([7, 5, 3, 1], 'B')
pointC = PointGP3([8, 1, 1, 1], 'C')

planeL = PlaneGP3(points=[pointA.vector,
                          pointB.vector, 
                          pointC.vector], name='L')

lineAB = LineGP3(points=[pointA.vector,
                         pointB.vector], name='AB')


print('El punto C esta en el plano? ', planeL.pointInPlane(pointC))
print('La linea L esta en el plano? ', planeL.lineInPlane(lineAB))

pointD = PointGP3([8, 3, 5, 1], 'D')
print('El punto D esta en el plano? ', planeL.pointInPlane(pointD))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pointA.plot(ax)
pointB.plot(ax)
pointC.plot(ax)
planeL.plot(ax)
lineAB.plot(ax)
pointD.plot(ax)
plt.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# ¿En que punto una línea atraviesa un plano?

lineL = LineGP3(points=[[-3, -6, 0, 1],
                        [1, 3,  5, 1]], name='L')

planePi = PlaneGP3(coefficients=[3, 1, 1, -10], name='Pi')
pointX = PointGP3(lineL.intersectionPlane(planePi).tolist(), 'X')

print('El punto X de interseccion es: ', pointX.vector)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lineL.plot(ax)
planePi.plot(ax)
pointX.plot(ax)

plt.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# ¿Dadas dos lineas L1 y L2, son estas lineas coplanares?

pointA = PointGP3([-2, -2, 0, 1], 'A')
pointB = PointGP3([7, 5, 3, 1], 'B')
pointBpri = PointGP3([7, 5, 0, 1], "B'")
pointC = PointGP3([8, 1, 1, 1], 'C')

lineAB = LineGP3(points=[pointA.vector,
                        pointB.vector], name='AB')

lineBC = LineGP3(points=[pointB.vector,
                        pointC.vector], name='BC')

lineBCpri = LineGP3(points=[pointC.vector,
                           pointBpri.vector], name="B'C")

print('Line AB is coplanar to BC?',lineAB.estimateCoplanar(lineBC))
print("Line AB is coplanar to B'C?",lineAB.estimateCoplanar(lineBCpri))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pointA.plot(ax)
pointB.plot(ax)
pointC.plot(ax)
pointBpri.plot(ax)

lineAB.plot(ax)
lineBC.plot(ax)
lineBCpri.plot(ax)

plt.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# ¿Dada una linea L y un plano Pi, cómo determinar si la linea L pertenece al plano Pi?
pointA = PointGP3([-2, -2, 0, 1], 'A')
pointB = PointGP3([7, 5, 3, 1], 'B')
pointC = PointGP3([8, 1, 1, 1], 'C')

lineAB = LineGP3(points=[pointA.vector,
                        pointB.vector], name='AB')

planeABC = PlaneGP3(points=[pointA.vector,
                            pointB.vector,
                            pointC.vector], name='ABC')

print(planeABC.lineInPlane(lineAB))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lineAB.plot(ax)
planeABC.plot(ax)
plt.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# ¿Cuál es la cónica C que resulta de intersectar un plano Pi con un elipsoide?
quadric = QuadricGP3(coeffs=[1/100, 1/16, 1/25, -1, 0, 0, 0, 0, 0, 0])

planePi = PlaneGP3(coefficients=[3, 1, 1, -10], name='Pi')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
quadric.plot(ax, xlim=(-10, 10), ylim=(-10, 10))
planePi.plot(ax, x_range=(-10, 10), y_range=(-5, 5))
planePi.plotIntersectionPlaneQuadric(ax, quadric, resolution=100)
ax.set_xlim([-10, 10])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
plt.show()