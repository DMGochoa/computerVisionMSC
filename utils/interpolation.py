import sys
sys.path.append("./")
import numpy as np
import math


class Interpolation:

    @staticmethod
    def get_pixel_value(img, x, y, method='bilinear'):
        """
        Get the pixel value using a specific interpolation method.
        """
        if method == 'nearest_neighbor':
            return img[min(int(round(y)), img.shape[0]-1), min(int(round(x)), img.shape[1]-1)]
        
        elif method == 'bilinear':
            x_floor, y_floor = int(x), int(y)
            dx, dy = x - x_floor, y - y_floor
            
            if x_floor + 1 < img.shape[1] and y_floor + 1 < img.shape[0]:
                top_left = img[y_floor, x_floor]
                top_right = img[y_floor, x_floor + 1]
                bottom_left = img[y_floor + 1, x_floor]
                bottom_right = img[y_floor + 1, x_floor + 1]

                interpolated = (1 - dx) * (1 - dy) * top_left + dx * (1 - dy) * top_right + (1 - dx) * dy * bottom_left + dx * dy * bottom_right
                return interpolated.astype(np.uint8)
            else:
                return img[y_floor, x_floor]

        elif method == 'bicubic':
            def bicubic_kernel(x):
                x = abs(x)
                if x <= 1:
                    return 1 - 2*x**2 + x**3
                elif x < 2:
                    return 4 - 8*x + 5*x**2 - x**3
                else:
                    return 0
            
            x_floor, y_floor = int(x), int(y)
            dx, dy = x - x_floor, y - y_floor
            
            interpolated = np.zeros(3)

            for m in range(-1, 3):
                for n in range(-1, 3):
                    if 0 <= x_floor + m < img.shape[1] and 0 <= y_floor + n < img.shape[0]:
                        weight = bicubic_kernel(m - dx) * bicubic_kernel(n - dy)
                        interpolated += weight * img[y_floor + n, x_floor + m]
            
            return np.clip(interpolated, 0, 255).astype(np.uint8)

        else:
            raise ValueError(f"Interpolation method {method} not recognized.")

    @staticmethod
    def nearest_neighbor(img, dstH, dstW):
        scrH, scrW, _ = img.shape
        ret_img = np.zeros((dstH, dstW, 3), dtype=np.uint8)
        for i in range(dstH):
            for j in range(dstW):
                scrX = round(i * (scrH / dstH))
                scrY = round(j * (scrW / dstW))
                ret_img[i, j] = img[min(scrX, scrH-1), min(scrY, scrW-1)]
        return ret_img

    @staticmethod
    def bilinear(image, dstH, dstW):
        scrH, scrW, channels = image.shape
        retimg = np.zeros((dstH, dstW, channels), dtype=np.uint8)

        for i in range(dstH):
            for j in range(dstW):
                scrx = (i + 1) * (scrH / dstH) - 1
                scry = (j + 1) * (scrW / dstW) - 1
                x = math.floor(scrx)
                y = math.floor(scry)
                u = scrx - x
                v = scry - y

                if x + 1 >= scrH or y + 1 >= scrW:
                    retimg[i, j] = image[x, y]
                else:
                    for c in range(channels):
                        retimg[i, j, c] = (1 - u) * (1 - v) * image[x, y, c] + \
                            u * (1 - v) * image[x + 1, y, c] + \
                            (1 - u) * v * image[x, y + 1, c] + \
                            u * v * image[x + 1, y + 1, c]

        return retimg

    @staticmethod
    def bicubic(img, dstH, dstW):
        scrH, scrW, _ = img.shape
        ret_img = np.zeros((dstH, dstW, 3), dtype=np.uint8)

        def bicubic_kernel(x):
            x = abs(x)
            if x <= 1:
                return 1 - 2*x**2 + x**3
            elif x < 2:
                return 4 - 8*x + 5*x**2 - x**3
            else:
                return 0

        for i in range(dstH):
            for j in range(dstW):
                scrX = i * (scrH / dstH)
                scrY = j * (scrW / dstW)
                x = int(scrX)
                y = int(scrY)

                tmp = np.zeros(3)
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        if (x+m) >= 0 and (x+m) < scrH and (y+n) >= 0 and (y+n) < scrW:
                            weight = bicubic_kernel(
                                scrX - x - m) * bicubic_kernel(scrY - y - n)
                            tmp += weight * img[x+m, y+n]

                ret_img[i, j] = np.clip(tmp, 0, 255).astype(np.uint8)

        return ret_img
