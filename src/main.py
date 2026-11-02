from ColorPicker import ColorPicker
from ColorPicker import pygame
from ColorPicker import pygame_widgets

cp = ColorPicker()

run = True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    cp.show_color_picker(cp.window)
    h, s, l, a = cp.get_hsla()
    
    pygame_widgets.update(event)
    pygame.display.update()
    
pygame.quit()
