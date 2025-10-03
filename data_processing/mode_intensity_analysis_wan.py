import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

# ================== Settings ==================
# Update this path to your image file
IMG_PATH = 'oam_2.jpg'

# Aperture settings (you can tune these)
R0_FACTOR = 0.3  # Ring radius as a fraction of the grid cell's smaller dimension
W_FACTOR = 0.2   # Ring half-width as a fraction of the grid cell's smaller dimension

# ===============================================

def get_charge_from_label(label):
    """Extracts the final integer topological charge from a string label for sorting."""
    if 'Center' in label:
        return 0
    # Find the number after the '=' sign
    match = re.search(r'(=)(-?\d+)', label)
    if match:
        return int(match.group(2))
    # Fallback for any labels that don't match
    return 999

# --- 1. Load & Background-Correct Image ---
# Load in color for visualization, and grayscale for processing
img_color = cv2.imread(IMG_PATH)
if img_color is None:
    raise FileNotFoundError(f"Image not found at path: {IMG_PATH}")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Use a large median blur to estimate the background glow and subtract it
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
bg = cv2.medianBlur(blur, 51)
corr = cv2.subtract(blur, bg).astype(np.float32)

# --- 2. Define the 3x3 Grid and Labels ---
H, W = corr.shape
ys = [0, H // 3, 2 * H // 3, H]
xs = [0, W // 3, 2 * W // 3, W]

# Labels corresponding to the theoretical positions in the 3x3 grid
grid_labels = [
    ['l_V-l_H=1',   'l_V=+2', 'l_H+l_V=3'],
    ['l_H=-1',      'Center (l=0)',  'l_H=+1'],
    ['-(l_H+l_V)=-3', 'l_V=-2', 'l_H-l_V=-1'],
]

# Precompute global coordinates for fast masking
Y, X = np.ogrid[:H, :W]

def sum_in_disc(cy, cx, r):
    """Calculates total intensity within a circular disc."""
    rsq = (X - cx)**2 + (Y - cy)**2
    mask = rsq <= r**2
    return float(corr[mask].sum())

def sum_in_annulus(cy, cx, r0, w):
    """Calculates total intensity within an annulus (ring)."""
    rsq = (X - cx)**2 + (Y - cy)**2
    mask = ((r0 - w)**2 <= rsq) & (rsq <= (r0 + w)**2)
    return float(corr[mask].sum())

# --- 3. Measure Intensity and Store Locations ---
mode_data = []
for i in range(3):
    for j in range(3):
        lbl = grid_labels[i][j]
        y0, y1 = ys[i], ys[i+1]
        x0, x1 = xs[j], xs[j+1]
        roi = corr[y0:y1, x0:x1]

        if np.any(roi > 0):
            r_y, r_x = np.unravel_index(np.argmax(roi), roi.shape)
            cy, cx = y0 + r_y, x0 + r_x
        else:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

        base = min(y1 - y0, x1 - x0)
        r0 = max(1, int(R0_FACTOR * base))
        w = max(1, int(W_FACTOR * base))

        if "Center" in lbl:
            r_disc = max(1, int(np.sqrt(4 * r0 * w)))
            s = sum_in_disc(cy, cx, r_disc)
        else:
            s = sum_in_annulus(cy, cx, r0, w)
        
        mode_data.append({
            'label': lbl,
            'intensity': max(0, s),
            'center': (cx, cy)
        })

# --- 4. Normalize Intensities ---
total_intensity = sum(m['intensity'] for m in mode_data)
if total_intensity > 0:
    for mode in mode_data:
        mode['normalized_intensity'] = (mode['intensity'] / total_intensity) * 100

# --- 5. Create Visualization ---
output_image = img_color.copy()
for mode in mode_data:
    cx, cy = mode['center']
    intensity_text = f"{mode['normalized_intensity']:.1f}%"
    
    # Draw a circle at the identified center and add labels
    cv2.circle(output_image, (cx, cy), 15, (0, 255, 0), 1) # Green circle
    cv2.putText(output_image, mode['label'], (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1) # Yellow label
    cv2.putText(output_image, intensity_text, (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1) # Cyan percentage

# --- 6. Plot Both Image and Histogram ---
# Sort the modes by topological charge for a clean histogram
mode_data.sort(key=lambda m: get_charge_from_label(m['label']))
plot_labels = [m['label'] for m in mode_data]
plot_vals = [m['normalized_intensity'] for m in mode_data]

# Create a figure with two subplots
plt.figure(figsize=(16, 6))

# Subplot 1: Annotated Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Identified OAM Mode Locations")
plt.axis('off')

# Subplot 2: Histogram
plt.subplot(1, 2, 2)
bars = plt.bar(plot_labels, plot_vals, color='darkcyan')
plt.ylabel("Normalized Intensity (%)")
plt.title("OAM Mode Intensity Distribution")
plt.xticks(rotation=60, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')

plt.tight_layout()
plt.show()