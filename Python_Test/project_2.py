"""INTEL REALSENSE DEPTH CAMERA D435i 颜色识别工程
    @Time   2023.10.24
    @Author SSC
"""

import pyrealsense2 as rs
import numpy as np
import cv2

#############################
# 基础设置段（Intel 标准例程）
#############################

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# 全局定义段
# 1. 饱和度增强定义
# 调节通道强度
lutEqual = np.array([i for i in range(256)]).astype("uint8")
lutRaisen = np.array([int(102+0.6*i) for i in range(256)]).astype("uint8")
# 调节饱和度
lutSRaisen = np.dstack((lutEqual, lutRaisen, lutEqual))  # Saturation raisen
# 2. 掩膜阈值定义
lower_green = np.array([30, 156, 58])
upper_green = np.array([66, 223, 232])
lower_red = np.array([117, 143, 0])
upper_red = np.array([133, 255, 255])
# 3. 结构元素定义
kernel = np.ones((9, 9), np.uint8)

#############################
# 运行段
#############################
try:
    while True:
        ##############################
        # 接收颜色图像数据
        ##############################
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        # cv2.imshow('color_image',color_image)
        # print(color_image.shape)

        """
            图像处理段
        """
        """
            1. 饱和度增强
        """
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)  # 色彩空间转换, RGB->HSV

        # cv2.imshow('hsv', hsv)
        blendSRaisen = cv2.LUT(hsv, lutSRaisen)             # 饱和度增大
        # img_enhance_saturation = cv2.cvtColor(blendSRaisen, cv2.COLOR_HSV2RGB)
        # cv2.imshow('img_enhance_saturation',img_enhance_saturation)
        """
            2. 掩膜创建
        """
        green_mask = cv2.inRange(blendSRaisen, lower_green, upper_green)
        green_img = cv2.bitwise_and(color_image, color_image, mask=green_mask)
        # cv2.imshow('result',green_img)
        red_mask = cv2.inRange(blendSRaisen, lower_red, upper_red)
        red_img = cv2.bitwise_and(color_image, color_image, mask=red_mask)
        # cv2.imshow('result',red_img)
        """
            3. 滤波
        """
        # green_img = cv2.GaussianBlur(green_img, (3, 3), 0)
        # red_img = cv2.GaussianBlur(red_img, (3, 3), 0)
        green_img = cv2.medianBlur(green_img, 5)
        red_img = cv2.medianBlur(red_img, 5)
        """
            4. 二值化
        """
        green_gray = cv2.cvtColor(green_img, cv2.COLOR_RGB2GRAY)
        red_gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)
        # green_gray = cv2.GaussianBlur(green_gray, (3, 3), 0)
        # red_gray = cv2.GaussianBlur(red_gray, (3, 3), 0)
        # cv2.imshow('result',green_gray)
        # green_thre = cv2.adaptiveThreshold(green_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,-10)
        # red_thre = cv2.adaptiveThreshold(red_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,-10)
        res, green_thre = cv2.threshold(
            green_gray, 0, 255, cv2.THRESH_BINARY)
        res, red_thre = cv2.threshold(
            red_gray, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow('result', green_thre)
        # cv2.imshow('result', red_thre)

        green_thre = cv2.morphologyEx(green_thre, cv2.MORPH_OPEN, kernel)
        red_thre = cv2.morphologyEx(red_thre, cv2.MORPH_OPEN, kernel)
        green_thre = cv2.morphologyEx(green_thre, cv2.MORPH_CLOSE, kernel)
        red_thre = cv2.morphologyEx(red_thre, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('result', red_thre)
        """
            5. Canny 提取边缘
        """
        red_edge = cv2.Canny(red_thre, 0, 255)

        circles = cv2.HoughCircles(red_thre, cv2.HOUGH_GRADIENT_ALT,
                                   1.5, 10, param1=50, param2=0.83, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cir_ in circles:
                cir = cir_[0]
                print('circle.shape:', cir.shape, 'circle:', cir)
                cv2.circle(
                    color_image, (cir[0], cir[1]), cir[2], (0, 255, 0), 2)
                cv2.circle(color_image, (cir[0], cir[1]), 2, (0, 0, 255), 3)

        cv2.imshow('result', color_image)

        key = cv2.waitKey(1)
        if key == 27:
            break


finally:

    # Stop streaming
    pipeline.stop()
