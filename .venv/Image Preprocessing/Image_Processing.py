from tkinter import Tk, filedialog
# import os
import cv2 as cv
import PIL as pl
import numpy as np
from matplotlib import pyplot as plt

# Temp method for testing this Program and a more convenient way of getting images
Tk().withdraw()

path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

# this is for pipelining the images into the CNN, which will be done later on.
#
# Image_folder = "../Images"
# image_files = [os.path.join(Image_folder, f)
#                for f in os.listdir(Image_folder)
#                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Reads if an image path and checks if the file actually contains an image
def image_read(path):
    src_img = cv.imread(path)
    assert src_img is not None, "File does not contain an image or file doesn't exist"
    return src_img

def image_alignment(src_img):
    gry_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    thrs_img = cv.threshold(gry_img, 0, 1, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)[1]
    coord = np.column_stack(np.where(thrs_img > 0))
    angle = cv.minAreaRect(coord)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gry_img.shape
    center = (w // 2, h // 2)
    matx = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, matx, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated


# Runs the Image into multiple processes to repair it for contouring
def image_processing(aln_img):
    gray_img = cv.cvtColor(aln_img, cv.COLOR_BGR2GRAY)

    blur_img = cv.GaussianBlur(gray_img, [3, 3], 0)

    lap = cv.Laplacian(blur_img, cv.CV_8U, ksize=1)

    ret, img_thrs = cv.threshold(lap, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return img_thrs

# Extracts the contour of the edges and draws it into the image for visual inspection and onto a 64 x 64 blank canvas for the CNN training
def contour_draw(img_thrs):
    cont, hierarchy = cv.findContours(img_thrs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cont_img = cv.cvtColor(img_thrs,cv.COLOR_GRAY2BGR)

    cv.drawContours(cont_img, cont, (-1), (0,255,0), 1)

    return cont_img

img = image_read(path)
aln_img = image_alignment(img)
proc_img = image_processing(aln_img)
res = contour_draw(proc_img)

kernal = np.ones((2, 2), np.uint8)
cls_img = cv.morphologyEx(res, cv.MORPH_CLOSE, kernal)

plt.figure(figsize=(cls_img.shape[1] / 100, cls_img.shape[0] / 100), dpi=100)
plt.imshow(cls_img)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()