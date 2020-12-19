import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
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
    return np.dstack([empty_pixels(img1) != empty_pixels(img2)]*3)

def empty_pixels(img):
    return (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)

def stitch(img1, img2):
    where_max = one_empty_pixel(img1, img2)
    stitched = 0.5 * img1 + 0.5 * img2
    stitched = stitched.astype(np.uint8)
    stitched = np.where(where_max, np.maximum(img1, img2), stitched)
    valid_points = np.argwhere(np.invert(empty_pixels(stitched)))
    top_left = valid_points.min(axis=0)
    bottom_right = valid_points.max(axis=0)
    return stitched[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SyntaxError('usage: stitch.py [path to first image] [path to second image]')
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    img1_ext = extend_image(img1, img2)
    img1_vert_padding = img1_ext[:,0:img1.shape[1],:]

    print('Detecting SIFT descriptors.')
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_vert_padding, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img1, None)
    n_matches = 100

    print('Selecting the best 100 descriptors.')
    bf = cv2.BFMatcher()
    n_matches = 100

    matches_for_plot = bf.match(des3, des2)
    matches_for_plot = sorted(matches_for_plot, key = lambda x:  x.distance)
    matches_for_plot = matches_for_plot[:n_matches]
    descriptors = cv2.drawMatches(img1, kp3, img2, kp2, matches_for_plot, None, flags=2)
    cv2.imwrite('descriptors.png', descriptors)

    matches = bf.match(des1, des2)
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

    print('Computing best transform with RANSAC.')
    start = time.time()
    T = transform.estimate_transformation(src, dst, th_dist = 5)
    print('Done, took {} seconds.'.format(time.time()-start))

    print('Warping second image, may take a few minutes.')
    start = time.time()
    img2_warped = transform.warp_image(img2, T, img1_ext.shape)
    print('Done, took {} seconds.'.format(time.time()-start))

    print('Stitching results.')
    start = time.time()
    stitched = stitch(img1_ext, img2_warped) 
    print('Done, took {} seconds.'.format(time.time()-start))

    cv2.imwrite('stitched.png', stitched)
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.show()
