import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx_abs = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely_abs = cv2.convertScaleAbs(sobely)

sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

plt.imshow(sobelx_abs, cmap='gray')
plt.show()

plt.imshow(sobely_abs, cmap='gray')
plt.show()

plt.imshow(sobel_combined, cmap='gray')
plt.show()


