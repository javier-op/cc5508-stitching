import cv2
from matplotlib import pyplot as plt
import numpy as np
import transform

def extend_image(img1, img2):
    h = img1.shape[0] + img2.shape[0]
    w = img1.shape[1] + img2.shape[1]
    extended_img1 = np.zeros((h, w, 3), dtype=np.uint8)
    i_start = int((h / 2) - (img1.shape[0] / 2))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            extended_img1[i_start+i][j] = img1[i][j]
    return extended_img1

def one_empty_pixel(img1, img2):
    selected = np.zeros(img1.shape, dtype=np.bool)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            selected[i][j] = [empty_pixel(img1[i][j]) != empty_pixel(img2[i][j])] * 3
    return selected

def empty_pixel(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0

def stitch(img1, img2):
    where_max = one_empty_pixel(img1, img2)
    stitched = 0.5 * img1 + 0.5 * img2
    stitched = stitched.astype(np.uint8)
    stitched = np.where(where_max, np.maximum(img1, img2), stitched)
    min_i = None
    for i in range(stitched.shape[0]):
        if not min_i:
            for j in range(stitched.shape[1]):
                if not empty_pixel(stitched[i][j]):
                    min_i = i
                    break
        else:
            break
    max_i = None
    for i in reversed(range(stitched.shape[0])):
        if not max_i:
            for j in range(stitched.shape[1]):
                if not empty_pixel(stitched[i][j]):
                    max_i = i
                    break
        else:
            break
    max_j = None
    for j in reversed(range(stitched.shape[1])):
        if not max_j:
            for i in range(stitched.shape[0]):
                if not empty_pixel(stitched[i][j]):
                    max_j = j
                    break
    return stitched[min_i:max_i, 0:max_j, :]



img1 = cv2.imread('./images/caso_1/1a.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./images/caso_1/1b.jpg', cv2.IMREAD_COLOR)
img1_ext = extend_image(img1, img2)
img1 = img1_ext[:,0:img1.shape[1],:]

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

n_matches = 100
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:  x.distance)
matches = matches[:n_matches]

src = np.zeros((3,n_matches), dtype=np.float32)
dst = np.zeros((3,n_matches), dtype=np.float32)
for i in range(n_matches) :
    src[0][i] = kp1[matches[i].queryIdx].pt[0]
    src[1][i] = kp1[matches[i].queryIdx].pt[1]
    src[2][i] = 1
    dst[0][i] = kp2[matches[i].trainIdx].pt[0]
    dst[1][i] = kp2[matches[i].trainIdx].pt[1]
    dst[2][i] = 1

T = transform.estimate_transformation(src, dst, th_dist = 5)
img2_warped = transform.warp_image(img2, T, img1_ext.shape)
stitched = stitch(img1_ext, img2_warped) 
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.show()
