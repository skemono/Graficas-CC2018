import numpy as np
import math


def noise(x, y, z, t=0):
    """Función de ruido simple usando seno"""
    return (math.sin(x * 12.9898 + y * 78.233 + z * 37.719 + t) * 43758.5453) % 1.0 - 0.5

def texturedShader(**kwargs):
    # Muestrear la textura usando UVs interpoladas
    u, v, w = kwargs["baryCoords"]
    A, B, C = kwargs["vertices"]
    texA = A[6:8] if len(A) > 7 else [0, 0]
    texB = B[6:8] if len(B) > 7 else [0, 0]
    texC = C[6:8] if len(C) > 7 else [0, 0]
    # Interpolar coordenadas UV
    tx = texA[0] * u + texB[0] * v + texC[0] * w
    ty = texA[1] * u + texB[1] * v + texC[1] * w
    texture = kwargs.get("texture", None)
    if texture:
        texColor = texture.get_color(tx, ty)
        r, g, b = [c / 255.0 for c in texColor]
    else:
        r, g, b = kwargs["pixelColor"]

    # Iluminación
    lighting_enabled = kwargs.get("lighting_enabled", True)
    ambient = kwargs.get("ambient", 0.2)
    if lighting_enabled:
        dirLight = kwargs["dirLight"]
        nA = [A[3], A[4], A[5]]
        nB = [B[3], B[4], B[5]]
        nC = [C[3], C[4], C[5]]
        normal = [
            nA[0] * u + nB[0] * v + nC[0] * w,
            nA[1] * u + nB[1] * v + nC[1] * w,
            nA[2] * u + nB[2] * v + nC[2] * w,
        ]
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = [normal[0] / normal_length, normal[1] / normal_length, normal[2] / normal_length]

        intensity = float(np.dot(normal, -np.array(dirLight)))
        intensity = max(0.0, min(1.0, intensity))
        intensity = max(ambient, intensity)

        r *= intensity
        g *= intensity
        b *= intensity

    # Limitar valores
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    return [r, g, b]

def vertexShader(vertex, normal, **kwargs):
    # Recibimos las matrices
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    viewportMatrix = kwargs["viewportMatrix"]
    current_time = kwargs.get("time", 0.0)  # Aceptar parámetro de tiempo
    normalMatrix = kwargs.get("normalMatrix", None)
    

    vt = [vertex[0], vertex[1], vertex[2], 1]
    
    nt = [normal[0], normal[1], normal[2], 0]
    
    vt = viewportMatrix @ projectionMatrix  @ viewMatrix  @ modelMatrix @ vt
    vt = vt.tolist()[0]
    
    # Transformar normal
    if normalMatrix is None:
        normalMatrix = np.linalg.inv(modelMatrix.T)
    nt = normalMatrix @ nt
    nt = nt.tolist()[0]
    
    vt = [vt[0] / vt[3], vt[1] / vt[3], vt[2] / vt[3]]
    nt = [nt[0], nt[1], nt[2]] 
    
    nt = nt / np.linalg.norm(nt)
    nt = nt.tolist()
    
    return vt,nt

def fragmentShader( **kwargs):
    r,g,b = kwargs["pixelColor"]
    return [r, g, b]

def flatShader(**kwargs):
    A,B,C = kwargs["vertices"]
    r,g,b = kwargs["pixelColor"]
    dirLight = kwargs["dirLight"]
    
    
    nA = [A[3],A[4],A[5]]
    nB = [B[3],B[4],B[5]]
    nC = [C[3],C[4],C[5]]
    
    normal = [ (nA[0] + nB[0] + nC[0]) / 3,
               (nA[1] + nB[1] + nC[1]) / 3,
               (nA[2] + nB[2] + nC[2]) / 3 ]

    # Normalizar el vector normal
    normal_length = np.linalg.norm(normal)
    if normal_length > 0:
        normal = normal / normal_length

    ambient = kwargs.get("ambient", 0.2)
    intensity = np.dot(normal, - np.array(dirLight))
    intensity = max(0, min(1, intensity)) 
    intensity = max(ambient, intensity)

    r *= intensity
    g *= intensity
    b *= intensity

    # Limitar valores finales de color para prevenir errores
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    return [r, g, b]


def gouraudShader(**kwargs):
    A, B, C = kwargs["vertices"]
    u, v, w = kwargs["baryCoords"]  
    r, g, b = kwargs["pixelColor"]
    dirLight = kwargs["dirLight"]
    
    nA = [A[3], A[4], A[5]]
    nB = [B[3], B[4], B[5]]
    nC = [C[3], C[4], C[5]]
    
    normal = [(nA[0] * u + nB[0] * v + nC[0] * w),
              (nA[1] * u + nB[1] * v + nC[1] * w),
              (nA[2] * u + nB[2] * v + nC[2] * w)]
    
    # Normalizar el vector normal interpolado
    normal_length = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if normal_length > 0:
        normal = [normal[0]/normal_length, normal[1]/normal_length, normal[2]/normal_length]
    
    ambient = kwargs.get("ambient", 0.2)
    intensity = np.dot(normal, -np.array(dirLight))
    intensity = max(0, min(1, intensity))  # Limitar entre 0 y 1
    intensity = max(ambient, intensity)

    r *= intensity
    g *= intensity
    b *= intensity
    
    # Limitación adicional de seguridad para los valores finales de color
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    return [r, g, b]

# ==== Helpers útiles para los shaders nuevos ====

def saturate(x): 
    return max(0.0, min(1.0, float(x)))

def lerp(a, b, t):
    t = saturate(t)
    return a * (1 - t) + b * t

def mix3(a, b, t):
    return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]

def hsv_to_rgb(h, s, v):
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return [r, g, b]

def fbm(x, y, z, t=0.0, octaves=4, lacunarity=2.0, gain=0.5):
    """Fractal Brownian Motion usando tu noise()."""
    amp = 0.5
    freq = 1.0
    total = 0.0
    for _ in range(octaves):
        total += amp * noise(x * freq, y * freq, z * freq, t * freq)
        freq *= lacunarity
        amp *= gain
    # Mapear [-~1, ~1] a [0,1]
    return saturate(total * 0.5 + 0.5)

def _interp_uv_and_color(kwargs, u, v, w, A, B, C):
    """Devuelve color base (r,g,b) ya sea de textura o de pixelColor."""
    texture = kwargs.get("texture", None)
    if texture:
        texA = A[6:8] if len(A) > 7 else [0, 0]
        texB = B[6:8] if len(B) > 7 else [0, 0]
        texC = C[6:8] if len(C) > 7 else [0, 0]
        tx = texA[0] * u + texB[0] * v + texC[0] * w
        ty = texA[1] * u + texB[1] * v + texC[1] * w
        r, g, b = [c / 255.0 for c in texture.get_color(tx, ty)]
    else:
        r, g, b = kwargs["pixelColor"]
    return [r, g, b]

def _interp_normal(kwargs, u, v, w, A, B, C):
    nA = [A[3], A[4], A[5]]
    nB = [B[3], B[4], B[5]]
    nC = [C[3], C[4], C[5]]
    nx = nA[0] * u + nB[0] * v + nC[0] * w
    ny = nA[1] * u + nB[1] * v + nC[1] * w
    nz = nA[2] * u + nB[2] * v + nC[2] * w
    nlen = (nx * nx + ny * ny + nz * nz) ** 0.5
    if nlen > 0:
        inv = 1.0 / nlen
        nx, ny, nz = nx * inv, ny * inv, nz * inv
    return [nx, ny, nz]

def _interp_pos(u, v, w, A, B, C):
    """Interpola posición (asumiendo A[0:3], B[0:3], C[0:3])."""
    px = A[0] * u + B[0] * v + C[0] * w
    py = A[1] * u + B[1] * v + C[1] * w
    pz = A[2] * u + B[2] * v + C[2] * w
    return [px, py, pz]


# ==========================================================
# 1) toonShader: sombreado tipo cómic + contorno por barycentrics
#    Params: levels=4, edge_width=0.06, outline_color=(0,0,0)
# ==========================================================
def toonShader(**kwargs):
    A, B, C = kwargs["vertices"]
    u, v, w = kwargs["baryCoords"]
    dirLight = kwargs["dirLight"]
    ambient = kwargs.get("ambient", 0.2)
    levels = int(kwargs.get("levels", 4))
    edge_width = kwargs.get("edge_width", 0.06)
    outline_color = kwargs.get("outline_color", (0.0, 0.0, 0.0))

    base = _interp_uv_and_color(kwargs, u, v, w, A, B, C)
    normal = _interp_normal(kwargs, u, v, w, A, B, C)

    # Intensidad luz difusa
    ndotl = max(0.0, float(-(normal[0]*dirLight[0] + normal[1]*dirLight[1] + normal[2]*dirLight[2])))
    ndotl = max(ambient, ndotl)

    # Cuantización tipo cel
    q = saturate(round(ndotl * (levels - 1)) / float(levels - 1))

    # Color toon
    color = [base[0] * q, base[1] * q, base[2] * q]

    # Contorno por proximidad a bordes del triángulo (barycentrics)
    edge = min(u, v, w)
    if edge < edge_width:
        color = list(outline_color)

    return [saturate(color[0]), saturate(color[1]), saturate(color[2])]


# =====================================================================
# 2) rimIridescentShader: rim lighting + iridiscencia (hue animado)
#    Params: rim_power=2.5, rim_strength=0.8, iridescent_sat=0.9
# =====================================================================
def rimIridescentShader(**kwargs):
    A, B, C = kwargs["vertices"]
    u, v, w = kwargs["baryCoords"]
    dirLight = kwargs["dirLight"]
    ambient = kwargs.get("ambient", 0.2)
    time = float(kwargs.get("time", 0.0))
    rim_power = float(kwargs.get("rim_power", 2.5))
    rim_strength = float(kwargs.get("rim_strength", 0.8))
    iridescent_sat = float(kwargs.get("iridescent_sat", 0.9))

    base = _interp_uv_and_color(kwargs, u, v, w, A, B, C)
    normal = _interp_normal(kwargs, u, v, w, A, B, C)

    # Lambert básico
    ndotl = max(0.0, float(-(normal[0]*dirLight[0] + normal[1]*dirLight[1] + normal[2]*dirLight[2])))
    lambert = max(ambient, ndotl)

    # Rim: idealmente usa viewDir, si no existe, aproxima con (0,0,1)
    viewDir = kwargs.get("viewDir", [0.0, 0.0, 1.0])
    vdotn = abs(float(normal[0]*viewDir[0] + normal[1]*viewDir[1] + normal[2]*viewDir[2]))
    rim = (1.0 - vdotn) ** rim_power

    # Iridiscencia: tono varía con la normal y el tiempo
    hue = (time * 0.12 + (normal[0] + normal[1] + normal[2]) * 0.17) % 1.0
    rim_rgb = hsv_to_rgb(hue, iridescent_sat, 1.0)

    # Composición: base iluminada + rim aditivo controlado
    lit = [base[0] * lambert, base[1] * lambert, base[2] * lambert]
    color = [
        saturate(lit[0] + rim_rgb[0] * rim * rim_strength),
        saturate(lit[1] + rim_rgb[1] * rim * rim_strength),
        saturate(lit[2] + rim_rgb[2] * rim * rim_strength),
    ]
    return color


# =====================================================================
# 3) lavaNoiseShader: lava procedural animada con fBm + glow leve
#    Params: scale=1.2, speed=0.8, contrast=1.6, glow=0.35
# =====================================================================
def lavaNoiseShader(**kwargs):
    A, B, C = kwargs["vertices"]
    u, v, w = kwargs["baryCoords"]
    dirLight = kwargs["dirLight"]
    ambient = kwargs.get("ambient", 0.2)
    time = float(kwargs.get("time", 0.0))
    scale = float(kwargs.get("scale", 1.2))
    speed = float(kwargs.get("speed", 0.8))
    contrast = float(kwargs.get("contrast", 1.6))
    glow = float(kwargs.get("glow", 0.35))

    # Colores de roca y lava
    rock = [0.08, 0.05, 0.03]
    lava = [1.00, 0.35, 0.05]

    # Posición interpolada en espacio objeto (asumido)
    px, py, pz = _interp_pos(u, v, w, A, B, C)

    # fBm animado
    n = fbm(px * scale, py * scale, pz * scale, t=time * speed, octaves=5)
    # Aumentar contraste y umbral para venas de lava
    veins = saturate((n - 0.45) * contrast + 0.5)

    # Mezcla roca-lava
    base = mix3(rock, lava, veins)

    # Iluminación difusa
    normal = _interp_normal(kwargs, u, v, w, A, B, C)
    ndotl = max(0.0, float(-(normal[0]*dirLight[0] + normal[1]*dirLight[1] + normal[2]*dirLight[2])))
    lambert = max(ambient, ndotl)

    # Glow/emisión leve en las partes más calientes
    emit = veins ** 2  # brillo más fuerte en zonas de alta vena
    color = [
        saturate(base[0] * lambert + emit * glow),
        saturate(base[1] * lambert + emit * glow * 0.8),
        saturate(base[2] * lambert + emit * glow * 0.5),
    ]
    return color


# ==============================================================================
# 4) wireframeOverlayShader: shading normal + malla alámbrica por barycentrics
#    Params: edge_width=0.03, line_color=(1,1,1), line_strength=1.0
# ==============================================================================
def wireframeOverlayShader(**kwargs):
    A, B, C = kwargs["vertices"]
    u, v, w = kwargs["baryCoords"]
    dirLight = kwargs["dirLight"]
    ambient = kwargs.get("ambient", 0.2)
    edge_width = kwargs.get("edge_width", 0.03)
    line_color = kwargs.get("line_color", (1.0, 1.0, 1.0))
    line_strength = float(kwargs.get("line_strength", 1.0))

    base = _interp_uv_and_color(kwargs, u, v, w, A, B, C)
    normal = _interp_normal(kwargs, u, v, w, A, B, C)
    ndotl = max(0.0, float(-(normal[0]*dirLight[0] + normal[1]*dirLight[1] + normal[2]*dirLight[2])))
    lambert = max(ambient, ndotl)
    lit = [base[0] * lambert, base[1] * lambert, base[2] * lambert]

    # Distancia a borde aproximada con barycentrics
    edge = min(u, v, w)
    t = 1.0 - saturate(edge / edge_width)  # 0 en interior, ~1 en borde
    if t <= 0.0:
        return [saturate(lit[0]), saturate(lit[1]), saturate(lit[2])]
    # Overlay de línea (aditivo limitado)
    color = [
        saturate(lit[0] + line_color[0] * t * line_strength),
        saturate(lit[1] + line_color[1] * t * line_strength),
        saturate(lit[2] + line_color[2] * t * line_strength),
    ]
    return color


