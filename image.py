#!/usr/bin/python3

import cv2
import numpy as np
import argparse

red_val = 0
x = 0
y = 0

font = cv2.FONT_HERSHEY_SIMPLEX

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output1 = image.copy()

height, width = output1.shape[:2]
crop = output1[y:y+np.round(3*height/5).astype("int"), x:x+width]
output = crop.copy()
h, w = output.shape[:2]

# Median blur used for noise reduction
blur_image = cv2.medianBlur(output, 5)

# Convert BGR to HSV
hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

# define range of RED color in HSV
red_lower = np.array([red_val-10, 85, 85])
red_upper = np.array([red_val+10, 255, 255])

Upper_red_lower = np.array([170-10, 85, 85])
Upper_red_upper = np.array([170+10, 255, 255])

# define range of GREEN color in HSV
green_lower = np.array([40, 50, 50])
green_upper = np.array([90, 255, 255])

Upper_green_lower = np.array([150, 65, 85])
Upper_green_upper = np.array([190, 255, 255])

# Threshold the HSV image to get only selected color
red_mask = cv2.inRange(hsv, red_lower, red_upper)
upper_red_mask = cv2.inRange(hsv, Upper_red_lower, Upper_red_upper)
final_maskr = cv2.add(red_mask, upper_red_mask)

# Threshold the HSV image to get only selected color
green_mask = cv2.inRange(hsv, green_lower, green_upper)
upper_green_mask = cv2.inRange(hsv, Upper_green_lower, Upper_green_upper)
final_maskg = cv2.add(green_mask, upper_green_mask)

# Bitwise-AND mask the original image
red_res = cv2.bitwise_and(output, output, mask=final_maskr)

# Bitwise-AND mask the original image
green_res = cv2.bitwise_and(output, output, mask=final_maskg)

# Convert to Black and White image
red_gray = cv2.cvtColor(red_res, cv2.COLOR_BGR2GRAY)

# Convert to Black and White image
green_gray = cv2.cvtColor(green_res, cv2.COLOR_BGR2GRAY)

# Laplacian for red color detection
dst_red = cv2.Laplacian(red_gray, cv2.CV_16S, 3)
abs_dst_red = cv2.convertScaleAbs(dst_red)

# Laplacian for green color detection
dst_green = cv2.Laplacian(green_gray, cv2.CV_16S, 3)
abs_dst_green = cv2.convertScaleAbs(dst_green)

# Hough circle detection for RED and GREEN
circles_red = cv2.HoughCircles(abs_dst_red, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=15, minRadius=3, maxRadius=25)
circles_green = cv2.HoughCircles(abs_dst_green, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=3, maxRadius=30)

print("Red Detection ")

if circles_red is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles_red = np.round(circles_red[0, :]).astype("int")
    foundVal = False

    circleNum_r = 0
    for (x_r, y_r, r_r) in circles_red:
        i_r = 0
        print("circleNum_r=", circleNum_r)
        print("x_r=", x_r, "y_r=", y_r, "Radius=", r_r)

        circleNum_r += 1
        for val_r_X in range(-3, 3):
            for val_r_Y in range(-3, 3):
                checkEdgeInfoIndex_r_y = y_r+val_r_Y
                checkEdgeInfoIndex_r_x = x_r+val_r_X
            # print(checkEdgeInfoIndex_gr)
                if checkEdgeInfoIndex_r_y < h and checkEdgeInfoIndex_r_y > 0 and checkEdgeInfoIndex_r_x < w and checkEdgeInfoIndex_r_x > 0:
                    edgeVal_r = crop[checkEdgeInfoIndex_r_y, checkEdgeInfoIndex_r_x]
#                    print("edgeVal=",edgeVal_r)
                    if (edgeVal_r[2] >= 200 and (edgeVal_r[1] < 175 or edgeVal_r[1] >= 200) and edgeVal_r[0] <= 70):
                        i_r += 1
#                        print("i_gr=",i_r)

                        if i_r > 10:
                            foundVal = True

                            cv2.circle(output1, (x_r, y_r), r_r, color=(0, 255, 0), thickness=2)
                            cv2.putText(output1, 'RED', (x_r, y_r), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            break
                if foundVal:
                    break
            if foundVal:
                foundVal = False
                break

        print(" ")

print("Green Detection ")

if circles_green is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles_green = np.round(circles_green[0, :]).astype("int")
    foundVal = False

    circleNum_gr = 0
    for (x_gr, y_gr, r_gr) in circles_green:
        i_gr = 0
        i_t = 0
        print("circleNum_gr=", circleNum_gr)
        print("x_gr=", x_gr, "y_gr=", y_gr, "Radius=", r_gr)

        circleNum_gr += 1
        for val_gr_X in range(-3, 3):
            for val_gr_Y in range(-3, 3):
                checkEdgeInfoIndex_gr_y = y_gr+val_gr_Y
                checkEdgeInfoIndex_gr_x = x_gr+val_gr_X
            # print(checkEdgeInfoIndex_gr)
                if checkEdgeInfoIndex_gr_y < h and checkEdgeInfoIndex_gr_y > 0 and checkEdgeInfoIndex_gr_x < w and checkEdgeInfoIndex_gr_x > 0:
                    edgeVal_gr = crop[checkEdgeInfoIndex_gr_y, checkEdgeInfoIndex_gr_x]
#                    print("edgeVal=",edgeVal_gr)
                    if (edgeVal_gr[1] >= 200 and edgeVal_gr[0] > 200 and (edgeVal_gr[2] > 100 and edgeVal_gr[2] < 220)):
                        i_t += 1
#                        print("i_gr=",i_gr)

                        if i_t > 10:
                            foundVal = True

                            cv2.circle(output1, (x_gr, y_gr), r_gr, color=(255, 0, 0), thickness=2)
                            cv2.putText(output1, 'GREEN', (x_gr, y_gr), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            break
                if foundVal:
                    break
            if foundVal:
                foundVal = False
                break

        print(" ")

cv2.imshow("output", output1)
# cv2.imwrite("Images/Results/image7.jpg", output1)
cv2.waitKey(0)
cv2.destroyAllWindows()
