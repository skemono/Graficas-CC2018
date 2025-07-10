import pygame

class Renderer(object): 
    def __init__(self, screen, bgColor=(0, 0, 0), currentColor=(255, 255, 255)):
        self.screen = screen 
        print(self.screen.get_size())
        self.width, self.height = self.screen.get_size()
        
        self.glColor(*(c/255 for c in currentColor)) 
        self.glClearColor(*(c/255 for c in bgColor)) 
        
        self.framebuffer = [[self.ClearColor for x in range(self.width)] 
                            for y in range(self.height)]
        
    def glClearColor(self, r, g, b): 
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))
        
        self.ClearColor = [r, g, b]
    
    def glColor(self, r, g, b):
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))
        
        self.currColor = [r, g, b]
    def glClear(self):
        color = [int(i*255) for i in self.ClearColor]
        self.screen.fill(color)
        
        self.framebuffer = [[self.ClearColor for x in range(self.width)] 
                            for y in range(self.height)]
        
    def glPoint(self, x, y, color=None):
        x = round(x)
        y = round(y)
        if (0 <= x < self.width and
            0 <= y < self.height):

            if color is None:
                color = [int(c*255) for c in self.currColor]
            elif len(color) == 3 and all(c <= 1 for c in color):
                color = [int(c*255) for c in color]
            
            self.screen.set_at((x, self.height - 1 - y), color)
            self.framebuffer[y][x] = [c/255 for c in color]
            
    def glLine(self,x0, y0, x1, y1): 
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx 
        
        if steep:
            x0, y0 = y0, x0 
            x1, y1 = y1, x1
            dx, dy = dy, dx
        
        if x0 > x1:
            x0, x1 = x1, x0 
            y0, y1 = y1, y0
            
        offset = 0
        threshold = dx 
        y = y0
        y_step = 1 if y0 < y1 else -1
        
        for x in range (x0, x1 +1 ): 
            if steep:
                self.glPoint(y, x) 
            else: 
                self.glPoint(x, y)
                
            offset += dy *2 
            if offset >= threshold: 
                y += y_step 
                threshold += dx * 2
    
    def glPolygonByCoords(self, coords, fillColor=None):
        if len(coords) < 3:
            return
        
        for i in range(len(coords)):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % len(coords)]
            self.glLine(x0, y0, x1, y1)
        
        if fillColor:
            min_x, max_x = min(x for x, y in coords), max(x for x, y in coords)
            min_y, max_y = min(y for x, y in coords), max(y for x, y in coords)
            
            for y in range(int(min_y), int(max_y) + 1):
                for x in range(int(min_x), int(max_x) + 1):
                    if self._pointInPolygon(x, y, coords):
                        self.glPoint(x, y, fillColor)
    
    def _pointInPolygon(self, x, y, coords):
        inside = False
        j = len(coords) - 1  
        
        for i in range(len(coords)):
            xi, yi = coords[i]      
            xj, yj = coords[j]     
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside 
            j = i
        
        return inside


