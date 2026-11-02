import pygame
import math
import pygame_widgets
from pygame_widgets.button import Button
'''
The ColorPicker class is responsible for selecting a color in RGB and converting it into HSL for the camera class.
This is done simple so we can select objects within a color range for the camera to detect for the AR4.

The following code was provided by a comment answer from stackoverflow and modified with the help of AI.

User: Rabbid76 is credited for this code snippet in response to another user asking for assistance with a color picker.

https://stackoverflow.com/questions/73517832/how-to-make-an-color-picker-in-pygame
'''


class ColorPicker:
    def __init__(self, x = 50, y = 50, w = 400, h = 60, p = 0):
        pygame.init()
        self.window = pygame.display.set_mode((500, 200))
        self.clock = pygame.time.Clock()
        self.clock.tick(60)
        
        self.rect = pygame.Rect(int(w * .2), int(w * .2), w, h)
        self.image = pygame.Surface((w, h))
        self.image.fill((0, 0, 0))
        #self.rad = h//2
        self.rad = int(w * .15)
        self.pwidth = w-self.rad*2
        self.p = p
        
        self.button_1 = CP_Button(self, self.window, 100, 50, "Color 1")
        self.button_2 = CP_Button(self, self.window, 100, 50, "Color 2")
        #The color picker's value ranges from 0 - 360. This iterates over each color and draws a rectangle for that color.
        #The dimensions can vary and can be adjust to user's preference using the pygame.draw.rect function.
        self.color_bar_width = 1
        self.color_bar_height = 25
        
        self.color_bar_top = ((self.rect.top + self.rect.bottom) // 2)
        self.color_bar_bottom = self.color_bar_top + self.color_bar_height
        
        for i in range(self.pwidth):
            color = pygame.Color(0)
            color.hsla = (int(360*i/self.pwidth), 100, 50, 100)
            pygame.draw.rect(self.image, color, (i + self.rad, 20, self.color_bar_width, self.color_bar_height), 200)
    
    #Gets mouse position         
    def update(self):
        moude_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        #Moves the position of the color selector based on where the mouse pointer is. The range is normalized between 0 and 1.
        if moude_buttons[0] and self.rect.collidepoint(mouse_pos):
            self.p = (mouse_pos[0] - self.rect.left - self.rad) / self.pwidth
            self.p = (max(0, min(self.p, 1)))
            
    def get_color(self):
        color = pygame.Color(0)
        color.hsla = (int(self.p * self.pwidth), 100, 50, 100)
        return color
    
    #Draw a circle that can be used to slide horizontally across the color picker.
    def draw(self, surf):
        surf.blit(self.image, self.rect)
        #Obtain the midpoint using the slider's top and bottom y values.
        circle_center = ((self.color_bar_top  + self.color_bar_bottom) // 2)
        #Place center is designated spot, the y value was obtained through trial and error.
        #10 was a good point for the center
        center = self.rect.left + self.rad + (self.p * self.pwidth), (circle_center) - 10
        #This circle acts as a border to see the slider better.
        pygame.draw.circle(surf, (0, 0, 0), center, (self.rect.height // 4)+ 2)
        pygame.draw.circle(surf, self.get_color(), center, self.rect.height // 4)
        
    #Conversion to HSLA for camera.
    def get_hsla(self):
        return self.get_color().hsla
    
    def show_color_picker(self, window: pygame.Surface):
        window.fill((30, 30, 30))
        
        self.update()
        self.draw(window)
        h, s, l, a = self.get_hsla()
        #Define font to display inside window using the pygame font system
        font = pygame.font.SysFont("Arial", 22)
        text = font.render(
            f"H: {round(h, 2)}  S: {s}  L: {l}",
            True,
            (255, 255, 255)
        )

        #The .blit function essentially stamps the text onto the "window" object creaded from pygame.
        #A better explanation of blit: https://stackoverflow.com/questions/37800894/what-is-the-surface-blit-function-in-pygame-what-does-it-do-how-does-it-work
        width, height = self.window.get_size()
        text_box = (int(width * .2), (height // 2 ) - 100)
        #.blit docs: https://www.pygame.org/docs/ref/surface.html#pygame.Surface.blit
        window.blit(text, text_box)
        
        #Following values were selected through trial and error to obtain a good look.
        self.button_1.button.setX(int(width * .2))
        self.button_1.button.setY((height // 2 ) + 50)
        self.button_1.button.draw()
        
        button_1_x = (self.button_1.button.getWidth() / 2) + self.button_1.button.getX()
        button_1_y = self.button_1.button.getY()
        
        self.button_2.button.setX(button_1_x + 100)
        self.button_2.button.setY(button_1_y)
        self.button_2.button.draw()
        
        pygame.display.flip()
    
class CP_Button:
    def __init__(self, cp : ColorPicker, source_window: pygame.Surface, width, height, name : str):
        self.cp = cp
        self.source_window = source_window
        self.width = width
        self.height = height
        
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
        color = pygame.Color(0)
        color.hsla =  self.cp.get_hsla()
        self.button.colour = color
        rgb = (color.r, color.g, color.b)

        self.button.inactiveColour = rgb
        self.button.hoverColour = rgb
        self.button.pressedColour = rgb
