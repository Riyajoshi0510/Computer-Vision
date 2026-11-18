
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Read image
image_path = Path(__file__).with_name("dog.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Could not read dog.jpg")

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.uint8)

# 3. Global thresholding
T = 127

global_threshold = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        if gray[i, j] > T:
            global_threshold[i, j] = 255
        else:
            global_threshold[i, j] = 0

# 4. Otsu thresholding

histogram = np.zeros(256, dtype=np.int64)

for pixel in gray.flatten():
    histogram[pixel] += 1

total_pixels = gray.size

total_sum = 0

for i in range(256):
    total_sum += i * histogram[i]

weight_background = 0
sum_background = 0

maximum_variance = 0
otsu_value = 0

for threshold in range(256):

    weight_background += histogram[threshold]

    if weight_background == 0:
        continue

    weight_foreground = total_pixels - weight_background

    if weight_foreground == 0:
        break

    sum_background += threshold * histogram[threshold]

    mean_background = sum_background / weight_background

    sum_foreground = total_sum - sum_background

    mean_foreground = sum_foreground / weight_foreground

    between_class_variance = (
        weight_background / total_pixels
        * weight_foreground / total_pixels
        * (mean_background - mean_foreground)**2
    )

    if between_class_variance > maximum_variance:

        maximum_variance = between_class_variance
        otsu_value = threshold

# Apply Otsu threshold
otsu_threshold = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        if gray[i, j] > otsu_value:
            otsu_threshold[i, j] = 255
        else:
            otsu_threshold[i, j] = 0

# 5. Adaptive Gaussian thresholding

block_size = 11
C = 2

radius = block_size // 2

padded = np.pad(
    gray,
    ((radius, radius), (radius, radius)),
    mode="reflect"
)

adaptive_threshold = np.zeros_like(gray)

# Create Gaussian kernel
sigma = 3

gaussian_kernel = np.zeros(
    (block_size, block_size),
    dtype=np.float64
)

for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):

        gaussian_kernel[
            x+radius, y+radius
        ] = np.exp(
            -(x**2+y**2)/(2*sigma**2)
        )

# Normalize kernel
gaussian_kernel = (
    gaussian_kernel / np.sum(gaussian_kernel)
)

# Apply adaptive threshold
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        region = padded[
            i:i+block_size,
            j:j+block_size
        ]

        local_mean = np.sum(
            region * gaussian_kernel
        )

        local_threshold = local_mean - C

        if gray[i, j] > local_threshold:
            adaptive_threshold[i, j] = 255
        else:
            adaptive_threshold[i, j] = 0

# 6. Print Otsu result
print("Otsu Optimum Threshold:", otsu_value)
print("Maximum Between-Class Variance:", maximum_variance)

# 7. Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Gaussian Threshold")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Experiment 9 completed successfully.")
