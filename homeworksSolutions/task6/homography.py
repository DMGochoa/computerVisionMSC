import sys
sys.path.append("./")
import os
import cv2
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from utils.interpolation import Interpolation
import matplotlib.pyplot as plt
import numpy as np


class HomographyEstimation:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.P, self.Q, _ = self.image.shape

    def show_image(self, image, title=""):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

    def compute_homography(self, src_pts, dst_pts):
        """
        Compute the homography matrix using selected correspondence points.
        """
        # Initialize the matrix A
        A = []

        for i in range(4):  # Since we have 4 correspondence points
            src = src_pts[i]
            dst = dst_pts[i]

            A.extend([
                [src[0], src[1], 1, 0, 0, 0, -dst[0]*src[0], -dst[0]*src[1]],
                [0, 0, 0, src[0], src[1], 1, -dst[1]*src[0], -dst[1]*src[1]]
            ])

        b = dst_pts.reshape(8)
        # Solve the system of equations
        h = np.linalg.lstsq(A, b, rcond=None)[0]
        # Reshape h to get the homography matrix
        H = np.vstack((h[:3], h[3:6], np.append(h[6:], [1])))
        return H

    def _apply_homography_section(self, section, H, interp_method):
        top, bottom, left, right = section
        dst_section = np.zeros((bottom-top, right-left, 3), dtype=np.uint8)
        inv_H = np.linalg.inv(H)

        for x in range(left, right):
            for y in range(top, bottom):
                src = inv_H @ np.array([x, y, 1])
                src = src / src[2]

                x_src, y_src = src[0], src[1]

                if 0 <= x_src < self.Q and 0 <= y_src < self.P:
                    dst_section[y-top, x-left] = Interpolation.get_pixel_value(
                        self.image, x_src, y_src, interp_method)

        return dst_section

    def apply_homography_manual(self, H, width, height, interp_method='bilinear', use_multithreading=False, thread_percentage=1):
        if not use_multithreading:
            return self._apply_homography_section((0, height, 0, width), H, interp_method)

        else:
            max_threads = int(thread_percentage * os.cpu_count())
            sections = self._split_image_sections(width, height, max_threads)
            dst = np.zeros((height, width, 3), dtype=np.uint8)

            with ThreadPoolExecutor(max_threads) as executor:
                future_to_section = {executor.submit(
                    self._apply_homography_section, section, H, interp_method): section for section in sections}
                for future in concurrent.futures.as_completed(future_to_section):
                    section = future_to_section[future]
                    top, bottom, left, right = section
                    dst[top:bottom, left:right] = future.result()

            return dst

    def _split_image_sections(self, width, height, num_sections):
        sqrt_sections = int(np.sqrt(num_sections))
        section_width = width // sqrt_sections
        section_height = height // sqrt_sections

        sections = []
        for i in range(sqrt_sections):
            for j in range(sqrt_sections):
                top = i * section_height
                bottom = (i+1) * \
                    section_height if i != sqrt_sections - 1 else height
                left = j * section_width
                right = (j+1) * section_width if j != sqrt_sections - \
                    1 else width
                sections.append((top, bottom, left, right))

        return sections

    def select_correspondence_points(self):
        """
        This function allows the user to manually select correspondence points 
        on the original image and a black image of the same size. The points will be
        displayed on the images after selection.
        """
        # Display original image and get points
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        original_points = plt.ginput(4)

        # Plot the selected points on the image
        for pt in original_points:
            plt.scatter(pt[0], pt[1], color='red')

        plt.show()

        # Clear the figure
        plt.close()

        # Display black image and get points
        black_image = np.zeros_like(self.image)
        plt.imshow(black_image)
        plt.title("Black Image")
        black_image_points = plt.ginput(4)

        # Plot the selected points on the black image
        for pt in black_image_points:
            plt.scatter(pt[0], pt[1], color='red')

        plt.show()

        # Clear the figure again
        plt.close()

        return original_points, black_image_points

    def apply_selected_homography(self, original_pts, black_image_pts, interp_method='bilinear', use_multithreading=False, thread_percentage=1):
        # Compute the homography
        H = self.compute_homography(
            np.array(original_pts), np.array(black_image_pts))
        # Apply the computed homography with specified interpolation method and multithreading options
        return self.apply_homography_manual(H, self.Q, self.P, interp_method, use_multithreading, thread_percentage)

import time
# Get the current directory path
dir_actual = os.path.dirname(os.path.abspath(__file__))
# Join the current directory path with your image's relative path
image_path = os.path.join(dir_actual, '..', 'task5', 'examples', 'cuadro.jpg')

homography = HomographyEstimation(image_path)
original_pts, black_image_pts = homography.select_correspondence_points()


# Sin multihilos
start_time = time.time()
transformed_image_no_threads = homography.apply_selected_homography(original_pts, black_image_pts, interp_method='bilinear', use_multithreading=False)
end_time = time.time()
print(f"Sin multihilos: {end_time - start_time:.4f} segundos")

# Con multihilos
start_time = time.time()
transformed_image_threads = homography.apply_selected_homography(original_pts, black_image_pts, interp_method='bilinear', use_multithreading=True, thread_percentage=0.5)
end_time = time.time()
print(f"Con multihilos (50% de hilos disponibles): {end_time - start_time:.4f} segundos")

plt.imshow(cv2.cvtColor(transformed_image_threads, cv2.COLOR_BGR2RGB))
plt.show()
