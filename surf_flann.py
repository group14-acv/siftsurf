import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img1 = cv2.imread('b_crop.png')
img2 = cv2.imread('bottle.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(400)

time_before = time.time()
kp1, des1 = surf.detectAndCompute(gray1, None)
kp2, des2 = surf.detectAndCompute(gray2, None)
time_after = time.time()


print('Time:', (time_after-time_before))

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

numOfMatches = 0
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        numOfMatches+=1

print("Number of matches:", numOfMatches)
print("Features found in img1:", len(kp1))
print("Features found in img2:", len(kp2))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.axis('off')
plt.imshow(img3,),plt.show()