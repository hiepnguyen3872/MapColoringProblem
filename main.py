import sys, copy, os, pygame
from pygame.locals import *
import constanst as MAPCOLORING
import tkinter as tk
from tkinter import filedialog
from preprocessImage import PreprocessImage
import cv2
import numpy as np
from mrv import run_mrv
from GeneticAlgorithm import GA_coloringmap
from csp_ac3 import Run

def main():
    global FPSCLOCK, DISPLAYSURF, ILLU, MAP, BASICFONT, BUTTONS, MAXSTATE, COLOURS

    COLOURS = ['red', 'green', 'blue', 'yellow']
    
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((MAPCOLORING.WINWIDTH, MAPCOLORING.WINHEIGHT))
    BUTTONS = [('Upload Image'), ('Genetic Algorithm'), ('Forward Checking'), ('ARC3')]
    MAXSTATE = len(BUTTONS) - 1
    pygame.display.set_caption('USA Coloring')
    BASICFONT = pygame.font.Font('assets/fonts/8-BITWONDER.TTF', 18)
    illu = pygame.image.load('assets/images/usa_map.png')
    width = illu.get_width()
    height = illu.get_height()
    scale = 0.2
    ILLU  = pygame.transform.scale(illu, (int(width * scale), int(height * scale)))
    root = tk.Tk()
    root.withdraw()

    while True:
        result = run()

def run():
    menuNeedsRedraw = True
    state = 0
    image = cv2.imread('assets/images/usa_map.jpg')
    preprocessedImage = None
    map = pygame.image.load('assets/images/usa_map.jpg')
    matrix = None
    mapFilled = None
    mapNeedsRedraw = True
    uploaded = True
    runAlgor = False

    while True: 
        cursorMoveTo = 0
        if uploaded == False and runAlgor == False:
            for event in pygame.event.get():
                if event.type == QUIT:
                    terminate()
                elif event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        
                        if state == 0:
                            temp = uploadPhoto()
                            if temp is not None:
                                image = temp
                                uploaded = True
                        else:
                            runAlgor = True
                        continue
                    elif event.key == K_UP:
                        cursorMoveTo = -1
                    elif event.key == K_DOWN:
                        cursorMoveTo = +1
                    elif event.key == K_ESCAPE:
                        terminate()

            if cursorMoveTo != 0:
                menuNeedsRedraw = True

        elif uploaded == True:
            DISPLAYSURF.fill(MAPCOLORING.WHITE)

            pleaseWaitSurf = pygame.Surface((MAPCOLORING.WINWIDTH, MAPCOLORING.WINHEIGHT))
            pleaseWaitSurf.fill(MAPCOLORING.BLUE)

            uploadingTxt, uploadingTxtRect = drawText('Image Preprocessing', 0, 0)
            uploadingTxtRect.center = ((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2))
            pleaseWaitSurf.blit(uploadingTxt, uploadingTxtRect)

            pleaseWaitTxt, pleaseWaitTxtRect = drawText('Please wait a few seconds', 0, 0)
            pleaseWaitTxtRect.center =((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2 + 30))
            pleaseWaitSurf.blit(pleaseWaitTxt, pleaseWaitTxtRect)

            pleaseWaitSurfRct = pleaseWaitSurf.get_rect()
            pleaseWaitSurfRct.topleft = (0,0)
            DISPLAYSURF.blit(pleaseWaitSurf, pleaseWaitSurfRct)

            pygame.display.update()
            preprocessedImage = PreprocessImage(image)
            preprocessedImage.img_2_matrix()
            matrix = preprocessedImage.get_adjacency_matrix()
            map = cv2ImageToSurface(preprocessedImage.image)

            mapNeedsRedraw = True
            uploaded = False

        elif runAlgor == True:
            DISPLAYSURF.fill(MAPCOLORING.WHITE)

            pleaseWaitSurf = pygame.Surface((MAPCOLORING.WINWIDTH, MAPCOLORING.WINHEIGHT))
            pleaseWaitSurf.fill(MAPCOLORING.BLUE)

            runningTxt, runningTxtRect = drawText(BUTTONS[state] + ' is being executed', 0, 0)
            runningTxtRect.center = ((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2))
            pleaseWaitSurf.blit(runningTxt, runningTxtRect)

            pleaseWaitTxt, pleaseWaitTxtRect = drawText('Please wait a few seconds', 0, 0)
            pleaseWaitTxtRect.center =((MAPCOLORING.WINWIDTH/2, MAPCOLORING.WINHEIGHT/2 + 30))
            pleaseWaitSurf.blit(pleaseWaitTxt, pleaseWaitTxtRect)

            pleaseWaitSurfRct = pleaseWaitSurf.get_rect()
            pleaseWaitSurfRct.topleft = (0,0)
            DISPLAYSURF.blit(pleaseWaitSurf, pleaseWaitSurfRct)

            pygame.display.update()
            
            if state == 1:
                output = GA_coloringmap(matrix, COLOURS)
            elif state == 2:
                output = run_mrv(matrix, COLOURS)
            elif state == 3:
                output = Run(matrix, COLOURS)
            
            if output is not None:
                output_image = preprocessedImage.colorize_map(output)
                map = cv2ImageToSurface(output_image)
            
            mapNeedsRedraw = True
            runAlgor = False
        
        DISPLAYSURF.fill(MAPCOLORING.WHITE)

        if menuNeedsRedraw:
            state = state + cursorMoveTo
            if state < 0:
                state = MAXSTATE
            elif state > MAXSTATE:
                state = 0
            menuSurf = drawMenu(state)
            menuNeedsRedraw = False

        menuSurfRect = menuSurf.get_rect()
        menuSurfRect.topleft = (0,0)
        DISPLAYSURF.blit(menuSurf, menuSurfRect)

        if mapNeedsRedraw:
            mapSurf = drawMap(map)
            mapNeedsRedraw = False
        mapSurfRect = mapSurf.get_rect()
        mapSurfRect.topleft = (MAPCOLORING.WINWIDTH*9/25,0)
        DISPLAYSURF.blit(mapSurf, mapSurfRect)
        
        pygame.display.update()
        FPSCLOCK.tick()

def uploadPhoto():
    file_path = filedialog.askopenfilename(filetypes=[('Image Files', ('.png', '.jpg'))])
    if file_path:
        image = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_UNCHANGED)
        return image
    else: 
        return None

def imgScale(image):
    width = image.get_width()
    height = image.get_height()
    scaleW = 1
    scaleH = 1
    if width > MAPCOLORING.WINWIDTH * (16/25):
        scaleW = MAPCOLORING.WINWIDTH * (16 / 25) / width
    if height > MAPCOLORING.WINHEIGHT:
        scaleH = MAPCOLORING.WINHEIGHT / height 
    scale = 1
    if scaleW > scaleH:
        scale = scaleH
    else:
        scale = scaleW
    
    scaledImg =  pygame.transform.scale(image, (int(width * scale), int(height * scale)))
    return scaledImg

def drawMap(map):
    image = imgScale(map)
    mapSurf = pygame.Surface((MAPCOLORING.WINWIDTH* (16/25), MAPCOLORING.WINHEIGHT))
    mapSurf.fill(MAPCOLORING.GREY)
    mapRect = image.get_rect()
    mapRect.center = (MAPCOLORING.WINWIDTH * (8 / 25) , MAPCOLORING.HALF_WINHEIGHT)
    mapSurf.blit(image, mapRect)
    return mapSurf

def drawMenu(state):
    menuSurf = pygame.Surface((MAPCOLORING.WINWIDTH * 9/25, MAPCOLORING.WINHEIGHT))
    menuSurf.fill(MAPCOLORING.BGCOLOR) 
    illustrationRect = ILLU.get_rect()
    illustrationRect.center = (175, 100)
    menuSurf.blit(ILLU, illustrationRect)
    nameSurface, nameRect = drawText('USA Coloring', 75, 250)
    menuSurf.blit(nameSurface, nameRect)

    xtop, ytop = 55, 300
    for x in range(len(BUTTONS)):
        btnSurface, btnRect = drawText(BUTTONS[x], xtop, ytop + 25 * x)
        menuSurf.blit(btnSurface, btnRect)

    cursorSurface, cursorRect = drawText('*', xtop - 25, ytop + 25 * state)
    menuSurf.blit(cursorSurface, cursorRect)

    return menuSurf

def drawText(text, x, y):
        textSurface = BASICFONT.render(text, True, MAPCOLORING.TEXTCOLOR)
        textRect = textSurface.get_rect()
        textRect.topleft = (x, y)
        return textSurface, textRect

def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image / 256).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
        format = 'RGB'
    else:
        format = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
        cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()

def terminate():
    pygame.quit()
    sys.exit()


    
if __name__ == '__main__':
    main()