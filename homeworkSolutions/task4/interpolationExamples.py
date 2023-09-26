import sys
sys.path.append("./")
from utils.interpolation import Interpolation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Get the current directory path
dir_actual = os.path.dirname(os.path.abspath(__file__))
# Join the current directory path with your image's relative path
ruta_video = os.path.join(dir_actual, 'examples', 'redpd.jpg')

image=np.array(Image.open(ruta_video))
print(image.shape)
# Mostrar la imagen original
plt.imshow(image)
plt.title("Original Image")

tamano = 800
tamano2 = 612

# Usar interpolación bilineal
bilinear_img = Interpolation.bilinear(image, tamano, tamano)
bilinear_img = Interpolation.bilinear(bilinear_img, tamano2, tamano2)

# Mostrar la imagen interpolada
plt.figure()
plt.imshow(bilinear_img)
plt.title("Bilinear Interpolation")

# Usar interpolación del vecino más cercano
nn_img = Interpolation.nearest_neighbor(image, tamano, tamano)
nn_img = Interpolation.nearest_neighbor(nn_img, tamano2, tamano2)

# Mostrar la imagen interpolada
plt.figure()
plt.imshow(nn_img)
plt.title("Nearest Neighbor Interpolation")

# Usar interpolación del vecino más cercano
bc_img = Interpolation.bicubic(image, tamano, tamano)
bc_img = Interpolation.bicubic(bc_img, tamano2, tamano2)

# Mostrar la imagen interpolada
plt.figure()
plt.imshow(bc_img)
plt.title("Bicubic Interpolation")

# Mostrar la imagen interpolada
plt.figure()
plt.imshow(np.clip(np.abs(image-bilinear_img), 0, 255))
plt.title("Diference")
plt.show()