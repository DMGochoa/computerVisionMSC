import sys
sys.path.append("./")
import os
import numpy as np
import time
import cv2
from utils.homography import HomographyEstimation
import matplotlib.pyplot as plt

# Get the current directory path
dir_actual = os.path.dirname(os.path.abspath(__file__))
# Join the current directory path with your image's relative path
image_path = os.path.join(dir_actual, '..', 'task5', 'examples', 'cuadro.jpg')
#image_path = os.path.join(dir_actual, 'capilla60.jpg')

homography = HomographyEstimation(image_path)
# Image points for the capilla60.jpg image
# [(132, 189),(443, 161),(131, 433),(425, 475)], [(100, 200),(460, 200),(100, 500),(460, 500)]
original_pts, black_image_pts = [(788, 271), (3169, 787), (595, 2320), (3480, 2626)], [(700, 500), (3280, 500), (700, 2460), (3280, 2460)]# homography.select_correspondence_points()
print(original_pts, black_image_pts)

N = 1  # Número de iteraciones
homography.select_correspondence_points()


# Usando OpenCV
opencv_times = []
print("Starting OpenCV")
for i in range(N):
    start_time = time.time()
    transformed_image_opencv = homography.apply_homography_opencv(original_pts, black_image_pts)
    end_time = time.time()
    opencv_times.append(end_time - start_time)
    print(f"OpenCV Time: {end_time - start_time:.4f} segundos | Iteration {i+1}")
# Sin multihilos
manual_times = []
print("Starting Manual")
for i in range(N):
    start_time = time.time()
    transformed_image_no_threads = homography.apply_selected_homography(original_pts,
                                                                        black_image_pts,
                                                                        interp_method='bilinear',
                                                                        use_multithreading=False)
    end_time = time.time()
    manual_times.append(end_time - start_time)
    print(f"Manual Time: {end_time - start_time:.4f} segundos | Iteration {i+1}")
# Con multihilos
parallel_times = []
for i in range(N):
    start_time = time.time()
    transformed_image_threads = homography.apply_selected_homography(original_pts,
                                                                     black_image_pts,
                                                                     interp_method='bilinear',
                                                                     use_multithreading=True,
                                                                     thread_percentage=0.2)
    end_time = time.time()
    parallel_times.append(end_time - start_time)
    print(f"Parallel Time: {end_time - start_time:.4f} segundos | Iteration {i+1}")

# Imprimir resultados
print(f"Manual Average Time: {sum(manual_times) / N:.4f} segundos mas o menos {np.std(manual_times):.4f}")
print(f"OpenCV Average Time: {sum(opencv_times) / N:.4f} segundos mas o menos {np.std(opencv_times):.4f}")
print(f"Parallel Average Time (50% de hilos disponibles): {sum(parallel_times) / N:.4f} segundos mas o menos {np.std(parallel_times):.4f}")


plt.figure()
plt.imshow(cv2.cvtColor(transformed_image_threads, cv2.COLOR_BGR2RGB))
plt.show()
