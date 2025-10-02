import cv2, numpy as np, matplotlib.pyplot as plt

# ================== settings (gentle) ==================
IMG_PATH = r"D:/MSc OQT/capstone/images/plus.jpg"

# Apertures (smaller than before)
R0_FACTOR = 0.36   # ring radius as fraction of cell min dim
W_FACTOR  = 0.10   # ring half-width fraction
AREA_NORMALISE = False  # True → mean brightness instead of total sum

# Display-only compression so "0" doesn't dominate (bars only)
GAMMA = 0.6        # 0.5–0.8 works well
# =======================================================

# --- Load & background-correct ---
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
assert img is not None, "Bad path: image not found."

blur = cv2.GaussianBlur(img, (5, 5), 0)
bg   = cv2.medianBlur(blur, 51)
corr = cv2.subtract(blur, bg).astype(np.float32)  # safe sums

H, W = corr.shape
ys = [0, H//3, 2*H//3, H]
xs = [0, W//3, 2*W//3, W]

labels = [
    ["+1", "+2", "+3"],
    ["-1", "0",  "+1b"],
    ["-3", "-2", "-1b"],
]

# Precompute global coords for fast masking
Y, X = np.ogrid[:H, :W]

def sum_and_area_disc(cy, cx, r):
    rsq = (X - cx)**2 + (Y - cy)**2
    m = rsq <= r*r
    return float(corr[m].sum()), int(m.sum())

def sum_and_area_annulus(cy, cx, r0, w):
    rsq = (X - cx)**2 + (Y - cy)**2
    m = ((r0 - w)**2 <= rsq) & (rsq <= (r0 + w)**2)
    return float(corr[m].sum()), int(m.sum())

# --- Measure each cell ---
intensities = {}
for i in range(3):
    for j in range(3):
        lbl = labels[i][j]
        y0, y1 = ys[i], ys[i+1]
        x0, x1 = xs[j], xs[j+1]
        roi = corr[y0:y1, x0:x1]

        # mode center = brightest pixel in the cell
        r_y, r_x = np.unravel_index(np.argmax(roi), roi.shape)
        cy, cx = y0 + r_y, x0 + r_x

        base = min(y1 - y0, x1 - x0)
        r0 = max(1, int(R0_FACTOR * base))
        w  = max(1, int(W_FACTOR * base))

        if lbl == "0":
            # Choose disc radius so its AREA ~ annulus area (fairer)
            r_disc = max(1, int(np.sqrt(4 * r0 * w)))
            s, a = sum_and_area_disc(cy, cx, r_disc)
        else:
            s, a = sum_and_area_annulus(cy, cx, r0, w)

        intensities[lbl] = (s / a) if (AREA_NORMALISE and a > 0) else s

# --- Plot with gentle gamma compression (display only) ---
order = ["+1", "+2", "+3", "-1", "0", "+1b", "-3", "-2", "-1b"]
vals  = np.array([intensities[k] for k in order], dtype=float)

plot_vals = np.power(np.clip(vals, 0, None), GAMMA)

plt.figure(figsize=(9, 5))
bars = plt.bar(order, plot_vals)
plt.ylabel(f"Integrated intensity (display γ={GAMMA})")
plt.title("Mode intensities")
plt.ylim(0, plot_vals.max() * 1.12)
plt.tight_layout()
plt.show()

# --- Print raw (un-gamma'd) ranking ---
print("Ranking (high→low) — RAW sums:")
for k, v in sorted(intensities.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{k:>4s}: {v:.1f}")
