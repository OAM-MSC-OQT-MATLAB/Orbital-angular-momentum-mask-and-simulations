# Create a smooth, colored 3D intensity surface like the example
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load and preprocess
path = r"D:/MSc OQT/capstone/images/Picture4.png"
img = Image.open(path).convert("L")

# Gentle Gaussian blur to reduce pixel noise so the 'hole' stands out
img_smooth = img.filter(ImageFilter.GaussianBlur(radius=1.5))

# Convert to normalized float
I = np.asarray(img_smooth, dtype=np.float64)
p1, p99 = np.percentile(I, [1, 99])     # robust contrast stretch
I = np.clip(I, p1, p99)
I = (I - I.min()) / (I.max() - I.min() + 1e-12)

# Downsample for a clean mesh (approx 250x250 grid)
h, w = I.shape
step = max(1, int(np.ceil(max(h, w) / 250)))
Z = I[::step, ::step]
ny, nx = Z.shape

# Centered coordinates scaled to ±0.05 (arbitrary units for a familiar look)
x = np.linspace(-0.05, 0.05, nx)
y = np.linspace(-0.05, 0.05, ny)
X, Y = np.meshgrid(x, y)

# Make a single 3D surface plot with a colorbar
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

ax.set_xlabel("x (arb. units)")
ax.set_ylabel("y (arb. units)")
ax.set_zlabel("Normalized intensity")
ax.set_title("3D Intensity Surface (smoothed & contrast-stretched)")

# Attach a colorbar
fig.colorbar(surf, ax=ax, shrink=0.7, label="Intensity")

plt.show()
# Re-plot with higher colour contrast using 'jet' colormap and clipping top values
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Clip top percentile to highlight variations inside bright region
Zc = np.minimum(Z, np.percentile(Z, 98))

surf = ax.plot_surface(X, Y, Zc, cmap="jet", linewidth=0, antialiased=True)

ax.set_xlabel("x (arb. units)")
ax.set_ylabel("y (arb. units)")
ax.set_zlabel("Normalized intensity")
ax.set_title("3D Intensity Surface (high contrast, clipped top)")

fig.colorbar(surf, ax=ax, shrink=0.7, label="Intensity")
plt.show()
