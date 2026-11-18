
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# 1. Read grayscale image
image = cv2.imread("hough.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not load hough.png")

# 2. Select CPU or GPU
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print("Using device:", device)

# 3. Convert image to tensor
img_tensor = torch.tensor(
    image, dtype=torch.float32, device=device
).unsqueeze(0).unsqueeze(0)

# 4. Define Sobel kernels
sobel_y = torch.tensor([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

sobel_x = sobel_y.transpose(-2, -1)

# 5. Apply padding
padded_img = torch.nn.functional.pad(
    img_tensor, (1, 1, 1, 1), mode='replicate'
)

# 6. Sobel convolution
detect_x = torch.nn.functional.conv2d(
    padded_img, sobel_x
)[0, 0]

detect_y = torch.nn.functional.conv2d(
    padded_img, sobel_y
)[0, 0]

# 7. Calculate gradient magnitude
resultant_tensor = torch.sqrt(
    detect_x**2 + detect_y**2
)

resultant = resultant_tensor.cpu().numpy()

plt.imshow(resultant, cmap="gray")
plt.title("Sobel Edge Detection")
plt.show()

# 8. Create Hough accumulator
height, width = resultant.shape

D = int(np.ceil(np.sqrt(width**2 + height**2)))

p_gpu = torch.zeros(
    (2*D + 1, 180),
    dtype=torch.int32,
    device=device
)

# 9. Find edge pixels
edge_y, edge_x = torch.where(resultant_tensor > 50)

# 10. Calculate theta
theta = torch.arange(180, dtype=torch.float32, device=device)
theta_rad = torch.deg2rad(theta)

cos_t = torch.cos(theta_rad)
sin_t = torch.sin(theta_rad)

# 11. Calculate rho and vote
if len(edge_y) > 0:

    x_coords = edge_x.unsqueeze(1)
    y_coords = edge_y.unsqueeze(1)

    rho = torch.round(
        x_coords*cos_t + y_coords*sin_t
    ).to(torch.int32) + D

    theta_indices = torch.arange(
        180, device=device
    ).unsqueeze(0).expand(rho.shape[0], -1)

    p_gpu.index_put_(
        (rho, theta_indices),
        torch.ones_like(rho),
        accumulate=True
    )

p = p_gpu.cpu().numpy()

print("Hough Accumulator created successfully")

# 12. Detect lines
threshold = 100

rho_idx_gpu, theta_idx_gpu = torch.where(
    p_gpu >= threshold
)

rho_idx = rho_idx_gpu.cpu().numpy()
theta_idx = theta_idx_gpu.cpu().numpy()

lines = []

for r, t in zip(rho_idx, theta_idx):
    rho_value = r - D
    theta_value = np.deg2rad(t)

    lines.append((rho_value, theta_value))

print("Detected lines:", len(lines))

# 13. Draw detected lines
output = cv2.cvtColor(
    resultant.astype(np.uint8),
    cv2.COLOR_GRAY2BGR
)

for rho, theta in lines:

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a*rho
    y0 = b*rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)

    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)

    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Hough Lines")
plt.axis("off")
plt.show()

# 14. Display accumulator
plt.figure(figsize=(12, 6))

plt.imshow(
    p,
    cmap="hot",
    aspect="auto",
    extent=[0, 180, D, -D]
)

plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.title("Hough Accumulator")
plt.colorbar(label="Votes")
plt.show()

# 15. Visualize parameter space
plt.figure(figsize=(10, 8))

edge_y, edge_x = torch.where(resultant_tensor > 50)

step = max(1, len(edge_x)//300)

edge_x_cpu = edge_x[::step].cpu().numpy()
edge_y_cpu = edge_y[::step].cpu().numpy()

m_range = np.linspace(-5, 5, 500)

for x, y in zip(edge_x_cpu, edge_y_cpu):
    c_vals = -x*m_range + y
    plt.plot(m_range, c_vals, color='cyan', alpha=0.15)

detected_m = []
detected_c = []

for rho, theta in lines:

    sin_t = np.sin(theta)

    if abs(sin_t) > 0.1:

        m = -np.cos(theta)/sin_t
        c = rho/sin_t

        if -5 <= m <= 5:
            detected_m.append(m)
            detected_c.append(c)

plt.scatter(
    detected_m,
    detected_c,
    color='magenta',
    s=25,
    label='Detected Lines'
)

plt.xlim(-5, 5)
plt.ylim(-1000, 2000)

plt.xlabel("Slope (m)")
plt.ylabel("Intercept (c)")
plt.title("Dual Space Line Intersections")
plt.grid(True)
plt.legend()
plt.show()

# 16. Find top 3 dominant lines
flat_indices = torch.argsort(
    p_gpu.flatten(),
    descending=True
)

top_rhos = []
top_thetas = []

visited = np.zeros_like(p)

for idx in flat_indices.cpu().numpy():

    r, t = np.unravel_index(idx, p.shape)

    r_start = max(0, r-10)
    r_end = min(p.shape[0], r+10)
    t_start = max(0, t-10)
    t_end = min(p.shape[1], t+10)

    if visited[r_start:r_end, t_start:t_end].sum() == 0:

        visited[r, t] = 1

        top_rhos.append(r-D)
        top_thetas.append(np.deg2rad(t))

        if len(top_rhos) == 3:
            break

# 17. Print top 3 lines
print("\nTop 3 Detected Lines:")

for idx, (rho, theta) in enumerate(
    zip(top_rhos, top_thetas)
):

    sin_t = np.sin(theta)

    if abs(sin_t) > 1e-6:

        m = -np.cos(theta)/sin_t
        c = rho/sin_t

        print(
            f"Line {idx+1}: "
            f"y = {m:.4f}x + {c:.4f}"
        )

    else:
        print(
            f"Line {idx+1}: Vertical line "
            f"(rho={rho})"
        )

print("\nExperiment 8 completed successfully.")
