#to plot intensities with detrending
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"D:/MSc OQT/capstone/images/oamsup.png"   


g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
g = (g - g.min()) / (g.max() - g.min() + 1e-12)

ksize = 51 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
top_hat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)

# --- intensity profile I(x) ---
gray = (top_hat - top_hat.min()) / (top_hat.max() - top_hat.min() + 1e-12)  # re-normalize
a = gray.mean(axis=0)                 # average over rows
x_pix = np.arange(a.size)


plt.plot(x_pix, a)
plt.xlabel("x (pixels)")
plt.ylabel("Intensity (a.u.)")


plt.show()
