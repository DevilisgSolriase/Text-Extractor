import cv2 as cv
import PIL as pl
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../Images/Untitled.png")
img2 = cv.imread("../Images/images.jpeg")
assert img is not None, "File does not contain an image or file doesn't exist"

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = cv.GaussianBlur(img, [5, 5], 0)

Lap = cv.Laplacian(img, cv.CV_8U, 1, 3)

ret, res = cv.threshold(Lap, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

cont, hierarchy = cv.findContours(Lap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

res = cv.cvtColor(res,cv.COLOR_GRAY2BGR)

cv.drawContours(res, cont, (-1), (0,255,0), 1)

res = cv.resize(res, dsize=(900,500))
# res1 = cv.resize(img2, None, fx=1.2, fy=1.2,interpolation=cv.INTER_AREA)

plt.imshow(res)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()