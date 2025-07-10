from gl import Renderer
import pygame

screenWidth = 800
screenHeight = round(screenWidth / 16 * 9)

bgColor = (0,0,0)
currentColor = (255, 255, 255)  


def main():
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    renderer = Renderer(screen, bgColor=bgColor, currentColor=currentColor)
    pygame.display.set_caption("Fill Polygon - June Herrera")
    clock = pygame.time.Clock()
    renderer.glClear()
    
    isRUNNING = True

    while isRUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRUNNING = False

        renderer.glPolygonByCoords([
        (165, 380), (185, 360), (180, 330),
        (207, 345), (233, 330), (230, 360),
        (250, 380), (220, 385), (205, 410),
        (193, 383)
        ], fillColor=(255, 140, 0))


        renderer.glPolygonByCoords([
            (321, 335), (288, 286), (339, 251),
            (374, 302)
        ], fillColor=(0, 200, 255))


        renderer.glPolygonByCoords([
            (377, 249), (411, 197), (436, 249)
        ], fillColor=(160, 60, 255))

        renderer.glPolygonByCoords([
            (413, 177), (448, 159), (502, 88),
            (553, 53), (535, 36), (676, 37),
            (660, 52), (750, 145), (761, 179),
            (672, 192), (659, 214), (615, 214),
            (632, 230), (580, 230), (597, 215),
            (552, 214), (517, 144), (466, 180)
        ], fillColor=(220, 200, 40))

        renderer.glPolygonByCoords([
            (682, 175), (708, 120), (735, 148),
            (739, 170)
        ], fillColor=bgColor)
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
    exit(0)


if __name__ == "__main__":
    main()