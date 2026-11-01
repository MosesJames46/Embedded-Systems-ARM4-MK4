
import cv2

'''
    The camera pyhton file uses the cv2 library from opencv to capture webcam frames.
    
    [USAGE]
    The camera's purpose is to solely return frames to be used later in processing.
    
    ------------------------------------------------------------------------------------------------------------------------------------------------
    Before a frame can be captured, we must first initialize a variable with a valid Video Captuer object. This object stores the 
    current webcam to be targeted during rendering. Once asserted, we must then check whether or not the frame has returned successfully. The
    video capture object has a .read() function which returns a tuple contianing a boolean and a MatLike structure for the frame.
    If the boolean is returned true, then we can process the frame using various options given by cv2.
    
    When finished with the frame, we MUST release and then destroy and windows if created.
    
    -------------------------------------------------------------------------------------------------------------------------------------------------
    
    Debugging:
        Do not stop a frame unless it is completely used, otherwise, error will throw.
'''


#The code below was outputted by AI and tested to ensure correct output. Code has also been adjust to align more with goals of project.


class Camera:
    #The camera index is the targeted webcam. The defualt webcam is 0. We can target others by inputting integers greater than 0.
    def __init__(self, camera_index):
        self.camera_index = camera_index
        #Capture holds data returned from obtaining webcam initialization.
        self.capture = None
        
    def start(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera.")
        
    #Check if the current frame was returned. If success, return the frame to be processed.
    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera has not been started.")
        
        success, frame = self.capture.read()
        
        if not success:
            return None
        
        return frame
    
    #Always release frame when.
    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
    #Utility function to show that frames have been successfully obtained.
    def show_frame(self):
        cv2.imshow("Original", self.read())
        
    def show_grayscale_frame(self):
        current_frame = self.read()
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray)
        
    def destroy_windows():
        cv2.destroyAllWindows()
        
camera = Camera(0)

camera.start()
while True:

    camera.show_frame()
    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.stop()
cv2.destroyAllWindows()
