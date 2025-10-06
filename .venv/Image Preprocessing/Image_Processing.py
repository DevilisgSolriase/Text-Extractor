from itertools import count
from tkinter import Tk, filedialog
# import os
import cv2 as cv
import PIL as pl
import numpy as np
from matplotlib import pyplot as plt


# 0 = scanned doc, 1 = phone image, 2 = textured/colored doc (e.g ID card)
image_type = 0

# Temp method for testing this Program and a more convenient way of getting images
Tk().withdraw()

path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

def doc_type(img):
    # per-pixel threshold
    t =15
    global image_type
    total_px = img.size

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur_img = cv.blur(gray_img, (31, 31))

    diff = abs(gray_img - blur_img)

    mean_abs_diff = np.mean(diff)

    pct_above_t = np.count_nonzero(diff > t) / total_px

    p90 = np.percentile(diff, 90)

    print(mean_abs_diff, pct_above_t, p90)

    if mean_abs_diff < 3 and pct_above_t < 0.02 :
        image_type = 0
    elif 3 <= mean_abs_diff <= 20 or 0.02 <= pct_above_t <= 0.10:
        image_type = 1
    elif mean_abs_diff > 10 or pct_above_t > 0.10 or p90 >40:
        image_type = 2

    print(image_type)
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

    scaled_img = cv.resize(gray_img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)

    plt.title("Resized Image")
    plt.imshow(scaled_img, cmap='gray')
    plt.show()

    blur_img = cv.GaussianBlur(scaled_img, (3,3), 0)

    plt.title('Blur Image')
    plt.imshow(blur_img, cmap='gray')
    plt.show()

    clahe = cv.createCLAHE(2.0, (8, 8))

    eqlhst_img = clahe.apply(blur_img)

    plt.title('Equalized Image')
    plt.imshow(eqlhst_img, cmap='gray')
    plt.show()

    img_thrs = cv.adaptiveThreshold(eqlhst_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 5)

    plt.title("Image Threshold")
    plt.imshow(img_thrs, cmap='gray')
    plt.show()

    cont_img = contour_draw(img_thrs)

    plt.title('Contour Image')
    plt.imshow(cont_img)
    plt.show()

    # kernal = np.ones((2, 2), np.uint8)
    # cls_img = cv.morphologyEx(cont_img, cv.MORPH_CLOSE, kernal)
    #
    # plt.title('Closed Image')
    # plt.imshow(cls_img)
    # plt.show()

    # lap = cv.Laplacian(blur_img, cv.CV_8U, ksize=3)
    #
    # ret, img_thrs = cv.threshold(lap, 1, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return cls_img

# Extracts the contour of the edges and draws it into the image for visual inspection and onto a 64 x 64 blank canvas for the CNN training
def contour_draw(img):
    cont, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cont_img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    cv.drawContours(cont_img, cont, (-1), (0,255,0), 1)

    return cont_img

img = image_read(path)
doc_type(img)
aln_img = image_alignment(img)
proc_img = image_processing(aln_img)

plt.title('Result')
plt.imshow(proc_img)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()