import sys
sys.path.append("./")
from geometry.conicalGP2 import ConicalGP2
from geometry.pointGP2 import PointGP2
from geometry.lineGP2 import LineGP2
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#TODO: PRUEBA DE LA LÍNEA Y EL PUNTO
pointP = PointGP2(array=[1, 2, 1],
                  name='P')
pointQ = PointGP2(array=[4, 5, 1],
                  name='Q')
lineL = LineGP2(array=[-3, 6, -3],
                name='L')
    
lineR = pointP.calcularLinea(pointQ, name='R')
print(lineR.vector)
pointX = lineL.calcularPunto(lineR, name='X')

plt.figure()
pointP.plot()
pointQ.plot()
lineR.plot()
lineL.plot()
pointX.plot()
plt.grid()
plt.legend()

# TODO: PRUEBA DE LA CÓNICA
array = [[1, 0, 0], [0, 1, 0], [0, 0, -25]]
conic = ConicalGP2(array, 'C')
# Verificar que la línea es tangente a la cónica
pointP = PointGP2(array=[4, 3, 1], name='P')
lineR = conic.calculateTanLine(pointP, 'R')
# Verificar que el punto de corte de linea tangente
lineR1 = LineGP2(array=[1, 0, -5], name='R1')
pointX = conic.calculatePointFromTanLine(lineR1, 'X')
print(conic.conic)
print(conic.element_name)
plt.figure()
# Conica
conic.plot()
# Tangente
pointP.plot()
lineR.plot()
# Punto de corte
lineR1.plot()
pointX.plot()
plt.grid()

puntos = [[0, 5], [5, 0], [0, -5], [-5, 0], [4, 3]]
conic = ConicalGP2(name='C', puntos=puntos)
plt.figure()
for punto, i in zip(puntos, range(len(puntos))):
    PointGP2(punto + [1], f'P{i}').plot()
conic.plot()
plt.grid()


f = lambda x: x**2 - 2
h = 100
x_real = np.linspace(-10, 10, h)
y_real = f(x_real)

x_ruido = x_real + np.random.normal(0, 0.1, h)
y_ruido = y_real + np.random.normal(0, 0.1, h)
puntos = np.stack([x_ruido, y_ruido], axis=1)
conic = ConicalGP2(name='C', puntos=puntos)
y = sp.Symbol('y')
print(conic.array)
print(conic.conic)
re = sp.solve(conic.conic, y)
print(re)
plt.figure()
for punto, i in zip(puntos, range(len(puntos))):
    PointGP2(punto.tolist() + [1], '').plot()
conic.plot(x = np.linspace(-10, 10, 800),y = np.linspace(-5, 100, 800))
#plt.plot(x_real, y_real, label='Real')
plt.grid()
plt.show()

import matplotlib.pyplot as plt

# Lista para almacenar las coordenadas de los puntos
points = []

def onclick(event):
    global points
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points.append((x, y))
        plt.scatter(x, y, c='red')
        plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.draw()

# Crear la figura y conectarla al evento de clic
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.grid(True)
plt.title('Interactive Point Plotter')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Conectar el evento de clic con la función onclick
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Asegúrate de cerrar la ventana de Matplotlib para continuar con el código.
# Una vez que hayas recopilado suficientes puntos (al menos 5 para satisfacer la condición en tu clase ConicalGP2), 
# puedes pasar esos puntos para encontrar la cónica que mejor se ajusta.

# Importar la clase ConicalGP2 (Asegúrate de que el archivo esté en la misma carpeta o ajusta la ruta según sea necesario)
# from tu_modulo import ConicalGP2

# Crear un objeto ConicalGP2
conic_fitted = ConicalGP2(puntos=points)

# Mostrar la ecuación de la cónica
print(f"Ecuación de la cónica: {conic_fitted.conic}")

# Trazar la cónica usando el método de la clase
conic_fitted.plot()

# Trazar los puntos recopilados
for point in points:
    plt.scatter(*point, c='red')
    plt.annotate(f'({point[0]:.2f}, {point[1]:.2f})', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

# Configurar el gráfico
plt.grid(True)
plt.title('Fitted Conic and Collected Points')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.show()

