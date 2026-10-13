import cv2
import numpy as np
import math

class HandDetector:
    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.5):
        pass

    def find_hands(self, img, draw=True):
        return img

    def get_finger_status(self, img):
        # 1. Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. Strict skin color range bounds
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 3. Filter layout masks
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        
        if cv2.contourArea(contour) < 5000:
            return None
            
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) <= 3:
            return None
            
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return None
            
        count_defects = 0
        
        for i in range(defects.shape[0]):
            # FIXED: Flatten or directly slice the inner elements safely
            s, e, f, d = defects[i].flatten()
            
            # Extract point matrix sets securely
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # Prevent Division by Zero if lengths are tiny
            if (2 * b * c) == 0:
                continue
                
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            
            if angle <= 90:
                count_defects += 1
                
        total_fingers = count_defects + 1
        
        # Complete mock structure return
        fingers_list = [1] * total_fingers + [0] * (5 - total_fingers)
        return fingers_list


