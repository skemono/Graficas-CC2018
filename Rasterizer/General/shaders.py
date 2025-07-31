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
    if lighting_enabled:
        dirLight = kwargs["dirLight"]
        nA = [A[3], A[4], A[5]]
        nB = [B[3], B[4], B[5]]
        nC = [C[3], C[4], C[5]]
        normal = [nA[0] * u + nB[0] * v + nC[0] * w,
                  nA[1] * u + nB[1] * v + nC[1] * w,
                  nA[2] * u + nB[2] * v + nC[2] * w]
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = [normal[0]/normal_length, normal[1]/normal_length, normal[2]/normal_length]
        intensity = np.dot(normal, -np.array(dirLight))
        intensity = max(0, min(1, intensity))
        r *= intensity
        g *= intensity
        b *= intensity

    # Limitar valores
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    return [r, g, b]
import numpy as np

def vertexShader(vertex, normal, **kwargs):
    # Recibimos las matrices
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    viewportMatrix = kwargs["viewportMatrix"]
    

    vt = [vertex[0], vertex[1], vertex[2], 1]
    
    nt = [normal[0], normal[1], normal[2], 0]
    
    vt = viewportMatrix @ projectionMatrix  @ viewMatrix  @ modelMatrix @ vt
    vt = vt.tolist()[0]
    
    # Transformar normal
    nt = np.linalg.inv(modelMatrix.T) @ nt
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

    intensity = np.dot(normal, - np.array(dirLight))
    intensity = max(0, min(1, intensity)) 

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
    
    intensity = np.dot(normal, -np.array(dirLight))
    intensity = max(0, min(1, intensity))  # Limitar entre 0 y 1

    r *= intensity
    g *= intensity
    b *= intensity
    
    # Limitación adicional de seguridad para los valores finales de color
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    return [r, g, b]