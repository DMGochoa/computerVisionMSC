"""This module demonstrates the creation and usage of conics class
on the P2 space.

Diego A Moreno G
MSc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from geometry.lineGP2 import LineGP2
from geometry.pointGP2 import PointGP2
from geometry.conicalGP2 import ConicalGP2

# Define a conic matrix
matrix = [[1, 0, 0], [0, 1, 0], [0, 0, -25]]
conic = ConicalGP2(matrix, 'C')

# Define a point and compute its tangent line
pointP = PointGP2(array=[4, 3, 1], name='P')
lineR = conic.calculateTanLine(pointP, 'R')

# Define another line and compute its intersection with the conic
lineR1 = LineGP2(array=[1, 0, -5], name='R1')
pointX = conic.calculatePointFromTanLine(lineR1, 'X')
print(conic.conic)
print(conic.element_name)

# Plotting
plt.figure()
conic.plot()
pointP.plot()
lineR.plot()
lineR1.plot()
pointX.plot()
plt.grid()
plt.title('Conic and Tangent Lines')
plt.pause(6)

# Using points to define a conic
points = [[0, 5], [5, 0], [0, -5], [-5, 0], [4, 3]]
conic = ConicalGP2(name='C', puntos=points)
plt.figure()
for point, i in zip(points, range(len(points))):
    PointGP2(point + [1], f'P{i}').plot()
conic.plot()
plt.grid()
plt.title('Conic Defined by 5 Points')
plt.pause(6)

# Fitting a conic to noisy data
f = lambda x: x**2 - 2
h = 100
x_real = np.linspace(-10, 10, h)
y_real = f(x_real)

std_noise = 0.2
x_noise = x_real + np.random.normal(0, std_noise, h)
y_noise = y_real + np.random.normal(0, std_noise, h)
points = np.stack([x_noise, y_noise], axis=1)
conic = ConicalGP2(name='C', puntos=points)
y = sp.Symbol('y')
solutions = sp.solve(conic.conic, y)

plt.figure()
for point, i in zip(points, range(len(points))):
    PointGP2(point.tolist() + [1], '').plot()
conic.plot(x=np.linspace(-10, 10, 800), y=np.linspace(-5, 100, 800))
plt.plot(x_real, y_real, label='True')
plt.grid()
plt.title('Fitting a Conic to Noisy Data')
plt.pause(6)
