"""

"""

import cv2
import os
import tkinter as tk

# Función para obtener dimensiones de la pantalla usando tkinter


def obtener_dimensiones_pantalla():
    root = tk.Tk()
    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()
    root.destroy()
    return ancho_pantalla, alto_pantalla


# Leer la imagen
# Obtiene el directorio del script actual
dir_actual = os.path.dirname(os.path.abspath(__file__))
# Ajusta según tu estructura de carpetas
ruta_imagen = os.path.join(dir_actual, 'examples', 'cuadro.jpg')
imagen = cv2.imread(ruta_imagen)

# Obtener dimensiones de la pantalla
ancho_pantalla, alto_pantalla = obtener_dimensiones_pantalla()

# Comparar las dimensiones de la imagen con las dimensiones de la pantalla
h, w, _ = imagen.shape
if w > ancho_pantalla or h > alto_pantalla:
    # Calcular factor de escala
    escala_w = ancho_pantalla / w
    escala_h = alto_pantalla / h
    escala = min(escala_w, escala_h)

    # Redimensionar la imagen
    dim = (int(w * escala), int(h * escala))
    imagen = cv2.resize(imagen, dim, interpolation=cv2.INTER_AREA)

# Mostrar la imagen
cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
