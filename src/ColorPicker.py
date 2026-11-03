import pygame
import cv2
import numpy as np
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider
'''
The ColorPicker class is responsible for selecting a color in RGB and converting it into HSL for the camera class.
This is done simple so we can select objects within a color range for the camera to detect for the AR4.

The following code was provided by a comment answer from stackoverflow and modified with the help of AI.

User: Rabbid76 is credited for this code snippet in response to another user asking for assistance with a color picker.

https://stackoverflow.com/questions/73517832/how-to-make-an-color-picker-in-pygame

------------------------------------------------------------------------------------------------------------------------

[How it works]
Just call 'show_color_picker' as it will return a tuple of two RGBA values from the color picker.
'''


class ColorPicker:
    def __init__(self, x = 50, y = 50, w = 400, h = 60, p = 0):
        pygame.init()
        self.window = pygame.display.set_mode((500, 400))
        self.clock = pygame.time.Clock()
        self.clock.tick(60)
        
        self.h_slider = Slider(
            self.window, 50, 50, 300, 20,
            min=0, max=179, initial=0
        )

        self.s_slider = Slider(
            self.window, 50, 100, 300, 20,
            min=0, max=255, initial=255
        )

        self.v_slider = Slider(
            self.window, 50, 150, 300, 20,
            min=0, max=255, initial=255
        )
        
        self.color_editor = Button(self.window, 400, 50, 100, 100)
        
        self.button_1 = CP_Button(self, self.window, 100, 50, "Color 1")
        self.button_2 = CP_Button(self, self.window, 100, 50, "Color 2")
        self.button_3 = CP_Button_Accept(self, 
                                         [self.button_1, self.button_2],
                                         self.window, 100, 50, "Accept")
        
    #Conversion to HSLA for camera.
    #def get_hsla(self):
        #return self.get_color().hsla
        
    def get_hsv(self):
        return (self.h_slider.getValue(), self.s_slider.getValue(), self.v_slider.getValue())
    
    def hsv_to_rgb(self):
        h, s, v = self.get_hsv()
        
        hsv = np.uint8([[[h, s, v]]])

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]

        rgb = tuple(int(value) for value in rgb)
        
        return rgb
    
    #The following functoin was provided partly by AI as a way of converting between HSV to RGB
    def update_color_editor(self):
        rgb = self.hsv_to_rgb()
        self.color_editor.colour = rgb
        self.color_editor.inactiveColour = rgb
        self.color_editor.hoverColour = rgb
        self.color_editor.pressedColour = rgb
        
    def show_color_picker(self, window: pygame.Surface):
        window.fill((30, 30, 30))
        
        self.h_slider.draw()
        self.s_slider.draw()
        self.v_slider.draw()
        self.update_color_editor()
        self.color_editor.draw()
        
        #self.draw(window)
        h, s, v= self.get_hsv()
        #Define font to display inside window using the pygame font system
        font = pygame.font.SysFont("Arial", 22)
        text = font.render(
            f"H: {round(h, 2)}  S: {s}  V: {v}",
            True,
            (255, 255, 255)
        )

        text_box = (50, 200)
        #.blit docs: https://www.pygame.org/docs/ref/surface.html#pygame.Surface.blit
        window.blit(text, text_box)
        
        #Following values were selected through trial and error to obtain a good look.
        self.button_1.button.setX(50)
        self.button_1.button.setY(250)
        self.button_1.button.draw()
        
        self.button_2.button.setX(150)
        self.button_2.button.setY(250)
        self.button_2.button.draw()
        
        self.button_3.button.setX(250)
        self.button_3.button.setY(250)
        self.button_3.button.draw()
        
        pygame.display.flip()
        if any(button.selected_color is None for button in self.button_3.array):
            return
        else:
            return ((self.button_1.selected_color, self.button_2.selected_color))
    
class CP_Button:
    def __init__(self, cp : ColorPicker, source_window: pygame.Surface, width, height, name : str):
        self.cp = cp
        self.source_window = source_window
        self.width = width
        self.height = height
        self.selected_color = None
        self.commited_color = None
        self.x, self.y = self.source_window.get_size()
        
        self.is_clicked = False
        
        self.button = Button(
            self.source_window,
            self.x, self.y,
            self.width, self.height,
            text = name,
            onClick = self.change_color
        )

    #Provided by AI and tested to ensure proper functionality
    def change_color(self):
        self.commited_color = self.cp.get_hsv()
        
        rgb = self.cp.hsv_to_rgb()
        
        self.button.colour = rgb
        self.button.inactiveColour = rgb
        self.button.hoverColour = rgb
        self.button.pressedColour = rgb
        return rgb
    
class CP_Button_Accept:
    def __init__(self, cp : ColorPicker, array : list[CP_Button] , source_window: pygame.Surface, width, height, name : str):
            self.cp = cp
            self.source_window = source_window
            self.width = width
            self.height = height
            
            self.array = array
            
            self.x, self.y = self.source_window.get_size()
            
            self.is_clicked = False
            
            self.button = Button(
                self.source_window,
                self.x, self.y,
                self.width, self.height,
                text = name,
                onClick = self.apply_changes
            )
            
    def apply_changes(self):
        for button in self.array:
            button.selected_color = button.commited_color

#Functional class used to create sliders for HSV inputs
class CP_Slider:
    def __init__(self, window : pygame.Surface, minimum, maximum, step_size):
        self.slider = Slider(window, 100, 100, 800, 40, min = minimum, max = maximum, step = step_size)
        
    

            
    