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
    
    def get_transformed_corners(self, H):
        # Esquinas de la imagen original
        corners = np.array([
            [0, 0, 1],
            [self.Q-1, 0, 1],
            [self.Q-1, self.P-1, 1],
            [0, self.P-1, 1]
        ])

        # Transformando las esquinas
        transformed_corners = []
        for corner in corners:
            new_corner = H @ corner
            new_corner /= new_corner[2]  # Normalizar
            transformed_corners.append(new_corner[:2])

        transformed_corners = np.array(transformed_corners)
        
        # Min and Max X, Y values
        min_x = int(np.floor(transformed_corners[:, 0].min()))
        max_x = int(np.ceil(transformed_corners[:, 0].max()))
        min_y = int(np.floor(transformed_corners[:, 1].min()))
        max_y = int(np.ceil(transformed_corners[:, 1].max()))
        
        return min_x, max_x, min_y, max_y


    def _split_image_sections(self, width, height, num_sections):
        """
        Split the image into horizontal strips.
        """
        section_height = height // num_sections
        sections = []

        for i in range(num_sections):
            top = i * section_height
            bottom = (i+1) * section_height if i != num_sections - 1 else height
            sections.append((top, bottom, 0, width))

        return sections

    def _recursive_split(self, section, min_height):
        """
        Recursively split the image section into horizontal strips.
        """
        top, bottom, left, right = section
        if (bottom - top) <= min_height:
            return [section]

        middle = (top + bottom) // 2
        top_section = (top, middle, left, right)
        bottom_section = (middle, bottom, left, right)

        return self._recursive_split(top_section, min_height) + self._recursive_split(bottom_section, min_height)

    def apply_homography_manual(self, H, width, height, interp_method='bilinear', use_multithreading=False, thread_percentage=1, min_height=50):
        # If not using multithreading, apply homography on the entire image
        if not use_multithreading:
            return self._apply_homography_section((0, height, 0, width), H, interp_method)
        else:
            max_threads = int(thread_percentage * os.cpu_count())
            initial_sections = self._split_image_sections(width, height, max_threads)

            # Use recursive splitting to get the final sections
            sections = []
            for section in initial_sections:
                sections.extend(self._recursive_split(section, min_height))

            dst = np.zeros((height, width, 3), dtype=np.uint8)
            inv_H = np.linalg.inv(H)
            min_x, max_x, min_y, max_y = self.get_transformed_corners(H)

            def map_section_to_original(section):
                top, bottom, left, right = section
                dst_section = np.zeros((bottom-top, right-left, 3), dtype=np.uint8)

                for x in range(max(left, min_x), min(right, max_x+1)):
                    for y in range(max(top, min_y), min(bottom, max_y+1)):
                        src = inv_H @ np.array([x, y, 1])
                        src = src / src[2]
                        x_src, y_src = src[0], src[1]
                        if 0 <= x_src < self.Q and 0 <= y_src < self.P:
                            dst_section[y-top, x-left] = Interpolation.get_pixel_value(
                                self.image, x_src, y_src, interp_method)

                return dst_section

            with ThreadPoolExecutor(max_threads) as executor:
                future_to_section = {executor.submit(
                    map_section_to_original, section): section for section in sections}
                for future in concurrent.futures.as_completed(future_to_section):
                    section = future_to_section[future]
                    top, bottom, left, right = section
                    dst[top:bottom, left:right] = future.result()

            return dst

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

    def apply_homography_opencv(self, original_pts, black_image_pts):
        """
        Compute and apply the homography using OpenCV.
        """
        src_pts = np.array(original_pts, dtype=np.float32)
        dst_pts = np.array(black_image_pts, dtype=np.float32)
        
        # Compute the homography matrix using OpenCV
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        # Warp the source image
        transformed_image = cv2.warpPerspective(self.image, H, (self.Q, self.P))
        
        return transformed_image