import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('box.jpg')
img2 = cv2.imread('box_in_scene.png')
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

array = np.zeros(10)
# BFMatcher with default params
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2)


# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None)
plt.imshow(img3),plt.show()

