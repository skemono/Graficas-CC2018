import struct

class Texture:
    def __init__(self, path):
        self.path = path
        self.width = 0
        self.height = 0
        self.pixels = []
        try:
            self.read()
        except Exception as e:
            print(f"Error reading texture file {path}: {e}")

    def read(self):
        with open(self.path, "rb") as image:
            image.seek(10)
            header_size = struct.unpack('=l', image.read(4))[0]
            image.seek(18)
            self.width = struct.unpack('=l', image.read(4))[0]
            self.height = struct.unpack('=l', image.read(4))[0]
            image.seek(header_size)

            for y in range(self.height):
                row = []
                for x in range(self.width):
                    b = ord(image.read(1))
                    g = ord(image.read(1))
                    r = ord(image.read(1))
                    row.append([r/255, g/255, b/255])
                self.pixels.append(row)

    def get_color(self, u, v):
        if 0 <= u <= 1 and 0 <= v <= 1:
            x = int(u * (self.width - 1))
            y = int((1 - v) * (self.height - 1))  # Voltear coordenada V
            if 0 <= y < self.height and 0 <= x < self.width:
                color = self.pixels[y][x]
                return [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]
        return [255, 0, 255]  # Retornar magenta para errores/fuera de límites
