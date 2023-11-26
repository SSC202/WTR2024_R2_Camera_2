import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('coin.jpg')
if img is None:
    print('Could not open or find the image ')
    exit(0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("threshold", thresh)    #阈值处理后会有紧挨着（粘连）的情况
# 去噪处理
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)  # 开运算
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)  # 膨胀操作

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # 距离背景点足够远的点认为是确定前景

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)  # 确定未知区域：减法运算
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)  # 设定坝来阻止水汇聚

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
plt.imshow(img)
plt.show()
