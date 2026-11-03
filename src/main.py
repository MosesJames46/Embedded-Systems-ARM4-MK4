from ColorPicker import ColorPicker
from ColorPicker import pygame
from ColorPicker import pygame_widgets
from ColorDetector import ColorDetector
from Camera import Camera
from Camera import cv2

CD = ColorDetector(ColorPicker(), Camera(0))

run = True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    CD.obtain_colors()
    #print(CD.cp.show_color_picker(CD.cp.window))
    #CD.cp.show_color_picker(CD.cp.window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pygame_widgets.update(event)
    pygame.display.update()
    
pygame.quit()

CD.camera.capture.release()
cv2.destroyAllWindows()