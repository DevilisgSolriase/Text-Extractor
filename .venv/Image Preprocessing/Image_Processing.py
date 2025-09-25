import cv2 as cv
import PIL as pl
import numpy as np
from matplotlib import pyplot as plt

src_img = cv.imread("../Images/Untitled.png")
assert src_img is not None, "File does not contain an image or file doesn't exist"

gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

blur_img = cv.GaussianBlur(gray_img, [3, 3], 0)

Lap = cv.Laplacian(blur_img, cv.CV_8U, ksize=1)

ret, img_thrs = cv.threshold(Lap, 1, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# plt.imshow(img_thrs)
# plt.show()

cont, hierarchy = cv.findContours(img_thrs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

res = cv.cvtColor(img_thrs,cv.COLOR_GRAY2BGR)

cv.drawContours(res, cont, (-1), (0,255,0), 1)

# plt.imshow(res)
# plt.show()

kernal = np.ones((2, 2), np.uint8)
cls_img = cv.morphologyEx(res, cv.MORPH_CLOSE, kernal)

cls_img = cv.resize(cls_img, dsize=(900,500))

plt.imshow(cls_img)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()