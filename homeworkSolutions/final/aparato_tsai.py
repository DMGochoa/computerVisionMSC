import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def process_images(image_paths, file_name=''):
    # Lista para almacenar los puntos recolectados de todas las imágenes
    all_collected_points = []

    def onclick(event, collected_points):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            collected_points.append([x, y, 1])
            plt.scatter(x, y, c='red')
            #plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            plt.draw()

    for idx, img_path in enumerate(image_paths):
        # Lista para almacenar los puntos recolectados de la imagen actual
        collected_points = []

        # Cargar y mostrar la imagen
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f'Click en la imagen: {file_name}{idx+1}')
        
        # Conectar el evento de clic con la función onclick
        fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, collected_points))

        plt.show()
        
        # Agregar los puntos recolectados de la imagen actual a la lista total
        if len(collected_points) > 0:
            all_collected_points.append(collected_points)

    return all_collected_points[0]

def automated_generator(x_axis=1.2,
                        y_axis=1.2,
                        z_axis=1.2,
                        no_x=5,
                        no_y=5,
                        no_z=5,
                        length_axis_x=3.8,
                        length_axis_y=3.8,
                        length_axis_z=3.8):
    plane_points =[]
    xy_starting_point = [x_axis, y_axis, 0]
    xz_starting_point = [x_axis, 0, z_axis]
    yz_starting_point = [0, y_axis, z_axis]
    # Llenado del plano x-z
    for i in range(no_z):
        for j in range(no_x):
            plane_points.append([round(xz_starting_point[0] + length_axis_x*j, 2),
                                 round(xz_starting_point[1], 2),
                                 round(xz_starting_point[2] + length_axis_z*i, 2)])
    # Llenado del plano y-z
    for i in range(no_z):
        for j in range(no_y):
            plane_points.append([round(yz_starting_point[0], 2),
                                 round(yz_starting_point[1] + length_axis_y*j, 2),
                                 round(yz_starting_point[2] + length_axis_z*i, 2)])
    # Llenado del plano x-y
    for i in range(no_y):
        for j in range(no_x):
            plane_points.append([round(xy_starting_point[0] + length_axis_x*j, 2),
                                 round(xy_starting_point[1] + length_axis_y*i, 2),
                                 round(xy_starting_point[2], 2)])
    return plane_points

def normalize(vec_points, scale):
    mean = np.mean(vec_points, axis=1)

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    puntos_normalizados = np.dot(T, vec_points)
    return T, puntos_normalizados.transpose()

def get_A_matrix(img_points, real_points):
    A = []
    for i in range(len(img_points)):
        x, y, _ = img_points[i]
        X, Y, Z = real_points[i]
        A.append([0, 0, 0, -X, -Y, -Z, y*X, y*Y, y*Z,])
        A.append([X, Y, Z, 0, 0, 0, -x*X, -x*Y, -x*Z])
        A.append([-y*X, -y*Y, -y*Z, x*X, x*Y, x*Z, 0, 0, 0])
    return np.array(A)


def dataFitting( A, round=3):
    """Performs Singular Value Decomposition (SVD) to fit the conic to the given points.

    Args:
        A (numpy.ndarray): The A matrix constructed from the points.
        round (int, optional): The number of decimal places to round the fitted parameters to. Defaults to 3.

    Returns:
        numpy.ndarray: The fitted parameters.
    """
    _, _, V = np.linalg.svd(A)
    V = V.T
    result = np.array([V[:, -1]])/V[-1, -1]
    return result.round(round)[0]

# Uso de la función
root_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(root_dir, 'imgs')
image_paths = [os.path.join(images_dir, f'tsai{i+1}.jpg') for i in range(7)]
image_points = np.array(process_images(image_paths))
# print(image_points)
# with open(os.path.join(root_dir,'aparato_tsai.json'), 'w') as f:
#     json.dump(image_points.tolist(), f)

with open(os.path.join(root_dir,'aparato_tsai.json'), 'r') as f:
    image_points = np.array(json.load(f))

real_points = np.array(automated_generator())

s2 = lambda vec: 2**0.5/(vec[:,0].mean()**2 + vec[:,1].mean()**2)**0.5
s3 = lambda vec: 3**0.5/(vec[:,0].mean()**2 + vec[:,1].mean()**2 + vec[:,2].mean()**2)**0.5


T, image_points_norm = normalize(image_points.transpose(), s2(image_points))
U, real_points_norm = normalize(real_points.transpose(), s3(real_points))

#print(image_points_norm)
#print(real_points_norm)

p_values = dataFitting(get_A_matrix(image_points_norm, real_points_norm))

p_matrix_norm = np.array([[p_values[0], p_values[3], p_values[6]],
                         [p_values[1], p_values[4], p_values[7]],
                         [p_values[2], p_values[5], p_values[8]]])

p_matrix = np.dot(np.dot(np.linalg.inv(T), p_matrix_norm), U)

K, R = np.linalg.qr(p_matrix)
print(p_matrix)

x_cam = np.dot(np.linalg.inv(p_matrix), image_points.transpose())
x_cam = x_cam.transpose()
x_cam = x_cam/x_cam[:,2].reshape(-1,1)
