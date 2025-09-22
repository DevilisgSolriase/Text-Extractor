import cv2 as cv
import PIL as pl
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../Images/Untitled.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "File does not contain an image or file doesn't exist"

img = cv.GaussianBlur(img, [3, 3], 0)
ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

plt.imshow(img, cmap='gray')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()