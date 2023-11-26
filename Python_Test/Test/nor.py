"""二值取反例程
    @Time   2023.10.22
    @Author SSC
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("input.jpg", flags=1)  # 读取彩色
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
res, thre = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
ret = cv.bitwise_not(thre)

while True:
    
    cv.imshow('img',thre)
    cv.imshow('res',ret)
    
    key = cv.waitKey(1)
    if key == 27:
        break
