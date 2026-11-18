import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)
ret, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.imshow(image, cmap='gray')
plt.show()

plt.imshow(thresh_binary, cmap='gray')
plt.show()


# 1. Adaptive Mean Thresholding
thresh_mean = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

# 2. Adaptive Gaussian Thresholding
thresh_gaussian = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

plt.imshow(image, cmap='gray')
plt.show()

plt.imshow(thresh_binary, cmap='gray')
plt.show()

plt.imshow(thresh_mean, cmap='gray')
plt.show()

plt.imshow(thresh_gaussian, cmap='gray')
plt.show()