#file has color filter

import cv2
import numpy as np

class ColorClassifier:
    def check_color(self, hsv, lower, upper):
        #check values in range
        mask = cv2.inRange(hsv, lower, upper)
        #set threshold num of pixels
        if cv2.countNonZero(mask)/mask.size > 0.05:
            return True
        return False

    # color filters
    def get_color(self, image):
        color = 'Silver'
        colorFound = False
        check_color = self.check_color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if not colorFound:
            blue_lower = (80, 40, 125)
            blue_upper = (120, 90, 225)
            if check_color(hsv, blue_lower, blue_upper):
                color = 'Blue'
                colorFound = True
        if not colorFound:
            red_lower = (150, 80, 75)
            red_upper = (180, 190, 205)
            if check_color(hsv, red_lower, red_upper):
                color = 'Red'
                colorFound = True
        if not colorFound:
            white_lower = (0, 0, 180)
            white_upper = (10, 20, 255)
            if check_color(hsv, white_lower, white_upper):
                color = 'White'
                colorFound = True
        if not colorFound:
            black_lower = (0, 0, 0)
            black_upper = (120, 50, 50)
            if check_color(hsv, black_lower, black_upper):
                color = 'Black'
                colorFound = True
        return color

    # blue - (90, 50, 125), (110, 75, 180)
    # blue - (80, 40, 125), (120, 90, 225)
    # black - (0, 0, 0), (120, 50, 50)
    # white - (0, 0, 180), (10, 20, 255)
    # red - (160, 100, 75), (180, 190, 205)
