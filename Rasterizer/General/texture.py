import struct


class Texture:
    def __init__(self, path):
        self.path = path
        self.width = 0
        self.height = 0
        self.pixels = []  # stored as 0..255 ints [r,g,b]
        self.read()

    def read(self):
        with open(self.path, "rb") as image:
            # Offsets in BMP header
            image.seek(10)
            data_offset = struct.unpack('<I', image.read(4))[0]
            image.seek(18)
            width = struct.unpack('<i', image.read(4))[0]
            height_raw = struct.unpack('<i', image.read(4))[0]
            top_down = height_raw < 0
            self.width = width
            self.height = abs(height_raw)

            # Move to pixel data
            image.seek(data_offset)
            # Rows are padded to multiples of 4 bytes
            row_stride = ((self.width * 3 + 3) // 4) * 4

            rows = []
            for _ in range(self.height):
                row_bytes = image.read(row_stride)
                row = []
                for x in range(self.width):
                    b = row_bytes[x * 3 + 0]
                    g = row_bytes[x * 3 + 1]
                    r = row_bytes[x * 3 + 2]
                    row.append([r, g, b])
                rows.append(row)

            # Normalize so row 0 is top
            if not top_down:
                rows.reverse()
            self.pixels = rows

    def get_color(self, u, v):
        # Expect OBJ-style UVs (v origin at bottom), so flip v
        if u is None or v is None or self.width <= 0 or self.height <= 0:
            return [255, 0, 255]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        x = int(u * (self.width - 1))
        y = int((1.0 - v) * (self.height - 1))
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y][x]
        return [255, 0, 255]
