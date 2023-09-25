import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

# Sección I
# Borra todo el entorno de trabajo
plt.close('all')

# Sección II
# Get the current directory path
dir_actual = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(dir_actual, 'capilla60.jpg')

# Lee la imagen de entrada y obtiene su tamaño.
InputImage = cv2.imread(image_path, cv2.IMREAD_COLOR)
InputImage = cv2.cvtColor(InputImage, cv2.COLOR_BGR2RGB)
P, Q, _ = InputImage.shape

plt.figure(1)
plt.imshow(InputImage)
plt.pause(0.5)

# Sección III
# Establecer 4 correspondencias
x1p = [135, 194]
x2p = [442, 162]
x3p = [426, 481]
x4p = [130, 437]

# Escoger una escala apropiada para el cuadrilátero rectificado
w = 100  # Valor en píxeles para el ancho del cuadrilátero
h = 130  # Valor en píxeles pare el alto del cuadrilátero

x1 = [0, 0]
x2 = [w, 0]
x3 = [w, h]
x4 = [0, h]

# Sección IV
# Calculo de la Homografía utilizando funciones de OpenCV
src_pts = np.array([x1p, x2p, x3p, x4p], dtype=np.float32)
dst_pts = np.array([x1, x2, x3, x4], dtype=np.float32)
H_cv, _ = cv2.findHomography(src_pts, dst_pts)

print("Matriz de Transformación Homogénea OpenCV:")
print(H_cv)

# Sección V
# Cálculo de la Homografía de forma manual

# Formar el sistema Ah = b
A = []
b = []

for i, (x, xp) in enumerate(zip([x1, x2, x3, x4], [x1p, x2p, x3p, x4p])):
    A.append([x[0], x[1], 1, 0, 0, 0, -xp[0]*x[0], -xp[0]*x[1]])
    A.append([0, 0, 0, x[0], x[1], 1, -xp[1]*x[0], -xp[1]*x[1]])
    b.append(xp[0])
    b.append(xp[1])

A = np.asarray(A)
b = np.asarray(b)

# Solución por mínimos cuadrados usando SVD
U, D, Vt = np.linalg.svd(A)
D_inv = np.diag(1 / D)
h = Vt.T @ D_inv @ U.T @ b


# Construir la matriz de homografía
H_manual = np.array([
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1]
])
print("\nMatriz de Transformación Homogénea Manual:")
print(H_manual)

# Sección VI
# Generar la nueva imagen y medir el tiempo
t0 = time.time()
OutOpenCV = cv2.warpPerspective(InputImage, H_cv, (Q, P))
tm = time.time() - t0

plt.figure(2)
plt.imshow(OutOpenCV)
plt.pause(0.5)


# Cálculo de las esquinas transformadas de la imagen original
corners = np.array([
    [0, 0, 1],
    [Q, 0, 1],
    [0, P, 1],
    [Q, P, 1]
])

transformed_corners = np.dot(H_manual, corners.T).T
transformed_corners /= transformed_corners[:, 2][:, np.newaxis]

# Encuentra los límites
x_min = int(np.floor(transformed_corners[:, 0].min()))
x_max = int(np.ceil(transformed_corners[:, 0].max()))
y_min = int(np.floor(transformed_corners[:, 1].min()))
y_max = int(np.ceil(transformed_corners[:, 1].max()))

# Sección VII
# "Mapping" de forma manual y medir tiempo
OutManual = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

# Función para evaluar la imagen original en una coordenada existente.
def eval_image(Q, image):
    return image[Q[1], Q[0], :]

# Función de interpolación bilineal sobre la imagen original.
def bilinear_interpolation(Pxy, Q11, Q12, Q21, Q22, image):
    return (
        eval_image(Q11, image) * (Q22[0] - Pxy[0]) * (Q22[1] - Pxy[1]) -
        eval_image(Q12, image) * (Q22[0] - Pxy[0]) * (Q11[1] - Pxy[1]) -
        eval_image(Q21, image) * (Q11[0] - Pxy[0]) * (Q22[1] - Pxy[1]) +
        eval_image(Q22, image) * (Q11[0] - Pxy[0]) * (Q11[1] - Pxy[1])
    )

# Función que verifica si el interpolante está dentro de
# los límites de la imagen original.
def is_inside_image(Q11, Q22, image_shape):
    return Q11[0] >= 0 and Q22[0] < image_shape[1] and Q11[1] >= 0 and Q22[1] < image_shape[0]

t0 = time.time()

for m in range(y_min, y_max):
    for n in range(x_min, x_max):
        coords = np.dot(H_manual, [n, m, 1])
        coords /= coords[2]

        # Interpolación bilineal.
        Pxy = [coords[0], coords[1]]
        Q11 = [int(np.floor(Pxy[0])), int(np.floor(Pxy[1]))]
        Q12 = [Q11[0], int(np.ceil(Pxy[1]))]
        Q21 = [int(np.ceil(Pxy[0])), Q11[1]]
        Q22 = [int(np.ceil(Pxy[0])), int(np.ceil(Pxy[1]))]

        # Verificar si la matriz interpolante existe dentro de la imagen original.
        if is_inside_image(Q11, Q22, InputImage.shape):
            OutManual[m - y_min, n - x_min, :] = bilinear_interpolation(Pxy, Q11, Q12, Q21, Q22, InputImage)

tp = time.time() - t0

plt.figure(3)
plt.imshow(OutManual)
plt.pause(0.5)

# Sección VIII
# Comparar tiempos y respuestas

print("\nTiempos:")
print(f"OpenCV: {tm:.4f} segundos")
print(f"Manual: {tp:.4f} segundos")

# Comparación entre OpenCV y manual
plt.figure(3)
difference = cv2.absdiff(OutOpenCV, OutManual)
plt.imshow(difference)
RMSerror = np.sqrt(np.mean(difference ** 2))
print(f"\nError Medio Cuadrático Total [RMS]: {RMSerror:.4f}")

plt.show()
