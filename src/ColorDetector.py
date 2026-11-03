from ColorPicker import ColorPicker
from ColorPicker import pygame
from pygame_widgets.button import Button
from Camera import Camera
from Camera import cv2
import numpy as np

class ColorDetector:
    def __init__(self, cp : ColorPicker, camera : Camera):
        self.cp = cp
        self.camera = camera
        self.camera.start()
        
    def obtain_colors(self):
        frame = self.camera.read()
        colors = self.cp.show_color_picker(self.cp.window)
        #Because colors may not yet be initialized, we check if a valid value was passed.
        if colors is None:
            return 
        color_1, color_2 = colors
        
        mask = self.mask_colors(color_1, color_2, frame)
        
        self.get_contours(mask, frame)
        
        
    def mask_colors(self, color_1, color_2, frame):
        #HSL values must be ordered from smallest to least.
        lower_color = np.minimum(color_1, color_2)
        upper_color = np.maximum(color_1, color_2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        return mask
    
    def rgb_to_hsv(self, rgb):
        rgb_array = np.uint8([[rgb]])

        hsv = cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2HSV
        )

        return hsv[0][0]
    
    def get_contours(self, mask, frame):
        contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
        )
        # Draw bounding boxes around detected objects
        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 500:

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
        
        cv2.imshow("Detection", frame)
        cv2.imshow("Mask", mask)

    cv2.waitKey(1)