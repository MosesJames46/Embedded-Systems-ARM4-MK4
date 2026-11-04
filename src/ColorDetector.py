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
        
        self.window_name = "Camera"
        
        #Data sent to the callback for updating color
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.roi_selected = False
        
        #This is to ensure that the color is persistent through iterations
        self.color_range = None
        
        # Tracking information
        self.previous_center = None
        self.tracked_center = None

        # Maximum distance an object can move between frames.
        self.max_tracking_distance = 100

        # Minimum contour area to consider a detection.
        self.min_contour_area = 500
        
        #Values set for necessary stable frames
        self.stable_frames = 0
        self.required_stable_frames = 5
    
    #callback util for activating selection on window. This gets set before calling any functions so that cv2 can handle input processing.
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi_end = (x, y)
            self.roi_selected = True
            
    def draw_slection(self, frame):
        if self.drawing and self.roi_start and self.roi_end:
            cv2.rectangle(
                frame,
                self.roi_start, 
                self.roi_end,
                (0, 255, 0),
                2
            )
        return frame
    
    def process_selection(self, frame):
        if not self.roi_selected:
            return None
        x1, y1  = self.roi_start
        x2, y2 = self.roi_end
        
        #This is done to ensure positive dimensions
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        if x_min == x_max or y_min == y_max:
            self.roi_selected = False
            return None
        
        hsv_frame = cv2.cvtColor(
            frame, 
            cv2.COLOR_BGR2HSV
        )
    
        hsv_roi = hsv_frame[y_min: y_max, x_min : x_max]
        color_range = self.estimate_color_range(hsv_roi)
        
        self.roi_selected = False
        return color_range
    
    '''
        Using the current frame, select an area using the mouse pointer to focus color picker.
        This color returned will be used to track object. 
    '''
    
    def obtain_colors(self):
            #Read current frame and convert it to HSV
            frame = self.camera.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            #Select portion of frame to use for color selecting
            x, y, w, h = self.get_area_of_intrest(frame)
            #Numpy slicing to select a range of values to be passed to the estimator.
            hsv_roi = hsv_frame[y:y+h, x:x+w]
            colors = self.estimate_color_range(hsv_roi)
            
            if colors is None:
                return
            color_1, color_2 = colors
            
            #Mask Colors
            mask = self.mask_colors(color_1, color_2, frame)
            mask = self.clean_mask(mask)
            
            self.get_contours(mask, frame)
        
    def obtain_color_v2(self):
        frame = self.camera.read()
        
        if frame is None:
            return
        
        frame = self.draw_slection(frame)
        
        color_range = self.process_selection(frame)
        if color_range is  not None:
            self.color_range = color_range
        
        display_frame = frame.copy()
        display_frame = self.draw_slection(display_frame)
        
        cv2.imshow(self.window_name, display_frame)
        
        #Detect using the previously saved color range
        if self.color_range is not None:
            color_1 , color_2 = self.color_range
            
            mask = self.mask_colors(color_1, color_2, frame)
            mask = self.clean_mask(mask)
            
            frame = self.get_contours(mask, frame)
            
            # Draw contours on a separate copy.
            contour_frame = frame.copy()
            contour_frame = self.get_contours(mask, contour_frame)
            
            
            cv2.imshow("Contour", contour_frame)
            cv2.imshow("Mask", mask)
        cv2.imshow(self.window_name, display_frame)
        
        
    def mask_colors(self, color_1, color_2, frame):
        #HSL values must be ordered from smallest to least.
        lower_color = np.minimum(color_1, color_2)
        upper_color = np.maximum(color_1, color_2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        return mask
    
    #This is an operation used to remove noise and fill in gaps that are caught by the mask colors option.
    #Code provided by AI
    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)

        # Remove small white regions.
        opening = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Fill small black regions.
        closing = cv2.morphologyEx(
            opening,
            cv2.MORPH_CLOSE,
            kernel
        )
        
        #cv2.imshow("Original Mask", mask)
        #cv2.imshow("After Opening", opening)
        #cv2.imshow("After Closing", closing)

        return closing
    
    def rgb_to_hsv(self, rgb):
        rgb_array = np.uint8([[rgb]])

        hsv = cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2HSV
        )

        return hsv[0][0]
    
    #Return the tracked center for the highlighted object
    def get_tracked_center(self):
        if self.tracked_center is None:
            return None
        
        return self.tracked_center
    
    def get_previouis_center(self):
            if self.previouis_center is None:
                return None
            
            return self.previouis_center
    
    def get_movement_error(self):
        if self.tracked_center is None or self.previous_center is None:
            return
        
        return tuple(np.subtract(self.previous_center, self.tracked_center))
    
    #The contour function draws a bounding box around the color selected in the frame.
    def get_contours(self, mask, frame):
        contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
        )
        
        #Array for detected objects
        detections = []
        
        # Draw bounding boxes around detected objects
        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            #Obtain the center of the bounding box for tracking
            cx = x + w // 2
            cy = y + h // 2

            detections.append({
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "area": area
            })

        #Check for any valid detections
        if not detections:
            self.tracked_center = None
            self.previous_center = None
            self.stable_frames = 0
            return frame
        
        #If this is our first detection, choose the largest object.
        if self.previous_center is None:
            target = max(detections, key = lambda obj: obj["area"])
            
        else:
            
            #Find the detection closest to the previous position by using the linagl.norm func to calculate the Euclidean dist between two points.
            target = min(
                detections,
                key = lambda obj : np.linalg.norm(
                    np.array(obj["center"]) - 
                    np.array(self.previous_center)
                )
            )
            
            #Check whether the closest detection is too far away
            distance = np.linalg.norm(
                np.array(target["center"]) -
                np.array(self.previous_center)
            )
            
            if distance > self.max_tracking_distance:
                self.tracked_center = None
                self.previous_center = None
                self.stable_frames = 0
                return frame
        
        #updates the tracking info
        self.tracked_center = target["center"]
        self.previous_center = self.tracked_center
        self.stable_frames += 1
        
        #Draw the tracked object's bounding box
        x, y, w, h = target["bbox"]
        cx, cy = self.tracked_center
        
        cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
        
        cv2.circle(
            frame, 
            (cx, cy),
            5, 
            (0, 0, 255),
            -1
        )
        
        #Display coords
        cv2.putText(
            frame, 
            f"Position: ({cx}, {cy})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        return frame
    
    #Will check against frames and if there is a tracked target on the screen
    def is_target_stable(self):
        return (
            self.tracked_center is not None
            and self.stable_frames >= self.required_stable_frames
        )
    
    #Return a selectable reigion on the canvas and use those coordinates to detect color value.
    def get_area_of_intrest(self, frame):
        roi = cv2.selectROI(
                    "Calibration",
                    frame,
                    showCrosshair=True,
                    fromCenter=False
                    )
        x, y, w, h = roi
        return (x, y, w, h)
    
    #This function was provided by AI.
    #The color function is used to isolate extremes within a selected color within the frame and clip the HSV values to the correct values.
    def estimate_color_range(self, hsv_roi):
        pixels = hsv_roi.reshape(-1, 3)

        # Ignore pixels with very low saturation.
        #This happens by selecting all pixels in the array and only accepting those values greater than 40
        pixels = pixels[pixels[:, 1] > 40]

        if len(pixels) == 0:
            return None

        # Calculate the lower and upper percentiles.
        lower = np.percentile(pixels, 5, axis=0)
        upper = np.percentile(pixels, 95, axis=0)

        # Add tolerance. This can be adjusted as needed to add or reduce tolerance
        lower -= np.array([2, 15, 15])
        upper += np.array([2, 15, 15])

        #This keeps values within the expected ranges of the HSV color channels
        lower = np.clip(lower, [0, 0, 0], [179, 255, 255])
        upper = np.clip(upper, [0, 0, 0], [179, 255, 255])

        #We MUST ensure that the values returned are integers. Otherwise errors will occur.
        return (
            lower.astype(np.uint8),
            upper.astype(np.uint8)
        )
        
    def calibrate_color(self):
        frame = self.camera.read()

        if frame is None:
            return

        roi = cv2.selectROI(
            "Calibration",
            frame,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, w, h = roi

        if w == 0 or h == 0:
            cv2.destroyWindow("Calibration")
            return

        hsv_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        hsv_roi = hsv_frame[y:y+h, x:x+w]

        color_range = self.estimate_color_range(hsv_roi)

        if color_range is not None:
            lower, upper = color_range

            print("Estimated lower HSV:", lower)
            print("Estimated upper HSV:", upper)
            
            return (color_range)

        cv2.destroyWindow("Calibration")