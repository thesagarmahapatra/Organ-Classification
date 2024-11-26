import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/Users/ssm/Code/FML Project/val_image_1002.png" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

hist = hist / hist.sum()

plt.figure(figsize=(8, 6))
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.grid()
plt.show()
