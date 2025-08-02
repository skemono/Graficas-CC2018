import numpy as np

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

def vertexShader(vertex, normal, **kwargs):
    # Recibimos las matrices
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    viewportMatrix = kwargs["viewportMatrix"]
    current_time = kwargs.get("time", 0.0)  # Aceptar parámetro de tiempo
    

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


def hologram_vertex_shader(vertex, normal, **kwargs):
    # Shader de vértice que crea distorsión holográfica y efectos de falla
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    viewportMatrix = kwargs["viewportMatrix"]
    current_time = kwargs.get("time", 0.0)

    # --- Lógica de distorsión holográfica ---
    
    vertex_array = np.array(vertex)
    
    # Crear efecto de línea de escaneo holográfica
    scan_line_freq = 15.0
    scan_line_speed = current_time * 3.0
    scan_line_wave = np.sin(vertex_array[1] * scan_line_freq + scan_line_speed)
    
    # Ruido digital para efecto de falla
    noise_seed = vertex_array[0] * 13.7 + vertex_array[1] * 27.3 + vertex_array[2] * 41.1 + current_time * 5.0
    glitch_noise = np.sin(noise_seed * 43.5) * np.cos(noise_seed * 67.2)
    
    # Aplicar distorsión holográfica
    distortion_strength = 0.03
    if abs(glitch_noise) > 0.8:  # Picos de falla aleatorios
        distortion_strength *= 3.0
    
    # Brillo holográfico - desplazamiento de vértices
    shimmer_x = np.sin(vertex_array[1] * 20.0 + current_time * 4.0) * distortion_strength
    shimmer_z = np.cos(vertex_array[0] * 25.0 + current_time * 3.5) * distortion_strength * 0.5
    
    # Efecto de descomposición digital
    breakdown_factor = abs(np.sin(current_time * 2.0 + vertex_array[2] * 8.0))
    if breakdown_factor > 0.9:
        # Desplazamiento aleatorio de vértices durante la descomposición
        breakdown_x = (glitch_noise - 0.5) * 0.1
        breakdown_y = (np.sin(noise_seed * 23.4) - 0.5) * 0.1
        breakdown_z = (np.cos(noise_seed * 31.7) - 0.5) * 0.1
        shimmer_x += breakdown_x
        shimmer_z += breakdown_z
        vertex_array[1] += breakdown_y
    
    # Aplicar desplazamiento de línea de escaneo
    scan_displacement = scan_line_wave * 0.02
    vertex_array[0] += shimmer_x + scan_displacement
    vertex_array[2] += shimmer_z

    vt = [vertex_array[0], vertex_array[1], vertex_array[2], 1]
    nt = [normal[0], normal[1], normal[2], 0]
    
    vt = viewportMatrix @ projectionMatrix @ viewMatrix @ modelMatrix @ vt
    vt = vt.tolist()[0]
    
    # Transformar normal
    nt = np.linalg.inv(modelMatrix.T) @ nt
    nt = nt.tolist()[0]
    
    vt = [vt[0] / vt[3], vt[1] / vt[3], vt[2] / vt[3]]
    nt = [nt[0], nt[1], nt[2]]
    
    nt_norm = np.linalg.norm(nt)
    if nt_norm > 0:
        nt = (np.array(nt) / nt_norm).tolist()
    
    return vt, nt


def hologramShader(**kwargs):
    # Shader holográfico futurista con líneas de escaneo y efectos digitales
    u, v, w = kwargs["baryCoords"]
    A, B, C = kwargs["vertices"]
    current_time = kwargs.get("time", 0.0)
    
    # Calcular posición del mundo
    posA = np.array(A[:3])
    posB = np.array(B[:3])
    posC = np.array(C[:3])
    world_pos = posA * u + posB * v + posC * w
    
    # === COLOR BASE DEL HOLOGRAMA ===
    
    # Color primario del holograma (cian-azul)
    base_color = np.array([0.0, 0.7, 1.0])
    
    # Color de acento secundario (verde brillante)
    accent_color = np.array([0.0, 1.0, 0.3])
    
    # === EFECTOS DE LÍNEAS DE ESCANEO ===
    
    # Líneas de escaneo horizontales
    scan_freq = 25.0
    scan_speed = current_time * 4.0
    scan_line = np.sin(world_pos[1] * scan_freq + scan_speed)
    scan_intensity = (scan_line + 1.0) * 0.5  # Normalizar a [0, 1]
    
    # Líneas de interferencia vertical
    interference_freq = 40.0
    interference = np.sin(world_pos[0] * interference_freq + current_time * 6.0)
    interference_intensity = (interference + 1.0) * 0.5
    
    # === RUIDO DIGITAL Y FALLAS ===
    
    # Generar ruido digital
    noise_seed = world_pos[0] * 15.7 + world_pos[1] * 23.1 + world_pos[2] * 31.3 + current_time * 8.0
    digital_noise = np.sin(noise_seed * 41.2) * np.cos(noise_seed * 67.8)
    
    # Efecto de falla - cambios súbitos de color
    glitch_factor = abs(np.sin(current_time * 3.0 + world_pos[2] * 12.0))
    glitch_active = glitch_factor > 0.85
    
    # === PATRÓN DE TRANSPARENCIA HOLOGRÁFICA ===
    
    # Crear transparencia holográfica con patrones en movimiento
    transparency_pattern1 = np.sin(world_pos[0] * 8.0 + current_time * 2.0)
    transparency_pattern2 = np.cos(world_pos[1] * 12.0 + current_time * 1.5)
    transparency_base = (transparency_pattern1 + transparency_pattern2) * 0.25 + 0.7
    
    # Efecto de parpadeo
    flicker = 0.9 + 0.1 * np.sin(current_time * 15.0)
    transparency = transparency_base * flicker
    
    # === CÁLCULOS DE ILUMINACIÓN ===
    
    # Obtener información de iluminación
    dirLight = kwargs.get("dirLight", (0, 0, -1))
    light_dir = -np.array(dirLight) / np.linalg.norm(np.array(dirLight))
    
    # Interpolar normal
    nA = np.array(A[3:6])
    nB = np.array(B[3:6])
    nC = np.array(C[3:6])
    normal = nA * u + nB * v + nC * w
    normal = normal / np.linalg.norm(normal)
    
    # Iluminación básica
    n_dot_l = max(0.1, np.dot(normal, light_dir))
    
    # === COMPOSICIÓN DE COLOR ===
    
    # Comenzar con color base del holograma
    final_color = base_color.copy()
    
    # Aplicar efectos de líneas de escaneo
    if scan_intensity > 0.7:
        # Líneas de escaneo brillantes
        final_color = final_color * 0.3 + accent_color * 0.7
    elif scan_intensity > 0.4:
        # Líneas de escaneo de intensidad media
        mix_factor = (scan_intensity - 0.4) / 0.3
        final_color = final_color * (1 - mix_factor * 0.4) + accent_color * mix_factor * 0.4
    
    # Aplicar interferencia
    if interference_intensity > 0.6:
        final_color *= 1.3  # Iluminar áreas de interferencia
    
    # Aplicar ruido digital
    noise_effect = digital_noise * 0.1
    final_color += noise_effect
    
    # Efectos de falla
    if glitch_active:
        # Corrupción aleatoria de canales de color
        if abs(digital_noise) > 0.5:
            final_color[0] = 1.0  # Corrupción roja
            final_color[1] *= 0.3
        elif abs(digital_noise) > 0.3:
            final_color[1] = 1.0  # Corrupción verde
            final_color[2] *= 0.5
        else:
            final_color[2] = 1.0  # Corrupción azul
            final_color[0] *= 0.4
    
    # Aplicar iluminación básica
    final_color *= n_dot_l
    
    # Realce de bordes del holograma
    edge_factor = abs(np.dot(normal, np.array([0, 0, 1])))
    if edge_factor < 0.3:  # Las áreas de los bordes brillan más
        final_color *= 1.5
    
    # Aplicar transparencia para efecto holográfico
    final_color *= transparency
    
    # Agregar pulsación sutil
    pulse = 0.8 + 0.2 * np.sin(current_time * 5.0)
    final_color *= pulse
    
    # === FINALIZACIÓN ===
    
    # Limitar valores
    r, g, b = final_color
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    return [r, g, b]
    


def rusty_vertex_shader(vertex, normal, **kwargs):
    # Shader de vértice agresivo que crea hoyos, erosión masiva y destrucción completa
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    viewportMatrix = kwargs["viewportMatrix"]
    current_time = kwargs.get("time", 0.0)
    
    # --- LÓGICA DE DESCOMPOSICIÓN Y DESTRUCCIÓN EXTREMA ---
    
    vertex_array = np.array(vertex)
    normal_array = np.array(normal)
    
    # === CREACIÓN DE HOYOS MASIVOS ===
    
    # Crear hoyos enormes que perforan el modelo
    hole_seed = vertex_array[0] * 11.3 + vertex_array[1] * 17.7 + vertex_array[2] * 23.1
    hole_pattern1 = np.sin(hole_seed * 2.0) * np.cos(hole_seed * 1.5)
    hole_pattern2 = np.cos(hole_seed * 3.0) * np.sin(hole_seed * 2.5)
    
    # Combinar patrones de hoyos para destrucción irregular
    combined_holes = (hole_pattern1 + hole_pattern2) / 2.0
    
    # Creación de hoyos EXTREMA - remoción masiva de material
    if combined_holes < -0.1:  # Umbral mucho mayor para hoyos más grandes
        hole_intensity = (combined_holes + 0.1) / 0.9  # Escalado más agresivo
        # Crear hoyos profundos moviendo vértices HACIA ADENTRO
        hole_depth = hole_intensity * 0.8  # Hoyos mucho más profundos
        vertex_array += normal_array * hole_depth
        
        # Agregar bordes dentados aleatorios a los hoyos
        jagged_x = np.sin(hole_seed * 20.0) * abs(hole_intensity) * 0.15
        jagged_y = np.cos(hole_seed * 25.0) * abs(hole_intensity) * 0.15
        jagged_z = np.sin(hole_seed * 30.0) * abs(hole_intensity) * 0.15
        vertex_array += np.array([jagged_x, jagged_y, jagged_z])
    
    # === EROSIÓN SUPERFICIAL CATASTRÓFICA ===
    
    # Crear patrones de erosión masiva
    erosion_seed = vertex_array[0] * 19.3 + vertex_array[1] * 31.7 + vertex_array[2] * 41.1
    
    # Múltiples escalas de destrucción
    erosion_massive = np.sin(erosion_seed * 1.5) * np.cos(erosion_seed * 1.0)
    erosion_large = np.sin(erosion_seed * 4.0) * np.cos(erosion_seed * 3.0)
    erosion_medium = np.sin(erosion_seed * 8.0) * np.cos(erosion_seed * 6.0)
    
    # Combinar para destrucción total
    total_erosion = (erosion_massive * 0.6 + erosion_large * 0.3 + erosion_medium * 0.1)
    
    # Aplicar erosión EXTREMA
    if total_erosion < -0.05:  # Umbral mucho más agresivo
        erosion_depth = (total_erosion + 0.05) * 0.4  # Erosión mucho más profunda
        vertex_array += normal_array * erosion_depth
    
    # === ASTILLAMIENTO Y DESMORONAMIENTO AGRESIVO ===
    
    # Crear trozos masivos que se caen
    chip_seed = vertex_array[0] * 47.3 + vertex_array[1] * 59.7 + vertex_array[2] * 71.1
    chip_pattern = np.sin(chip_seed * 8.0) * np.cos(chip_seed * 6.0)
    
    if abs(chip_pattern) > 0.4:  # Astillamiento mucho más frecuente
        chip_intensity = (abs(chip_pattern) - 0.4) / 0.6
        
        # Crear trozos irregulares MASIVOS
        chunk_x = np.sin(chip_seed * 15.0) * chip_intensity * 0.2  # Trozos 10x más grandes
        chunk_y = np.cos(chip_seed * 18.0) * chip_intensity * 0.2
        chunk_z = np.sin(chip_seed * 22.0) * chip_intensity * 0.2
        
        vertex_array += np.array([chunk_x, chunk_y, chunk_z])
        
        # Agregar colapso severo del material
        collapse_depth = chip_intensity * 0.3
        vertex_array += normal_array * (-collapse_depth)
    
    # === DESTRUCCIÓN SUPERFICIAL EXTREMA ===
    
    # Destruir completamente la integridad de la superficie
    destruction_seed = vertex_array[0] * 83.7 + vertex_array[1] * 97.1 + vertex_array[2] * 113.3
    
    # Múltiples patrones de destrucción
    destruction1 = np.sin(destruction_seed * 12.0) * 0.08  # Mucho más agresivo
    destruction2 = np.cos(destruction_seed * 20.0) * 0.06
    destruction3 = np.sin(destruction_seed * 35.0) * 0.04
    
    total_destruction = destruction1 + destruction2 + destruction3
    vertex_array += normal_array * total_destruction
    
    # === REDES DE GRIETAS MASIVAS ===
    
    # Crear sistemas de grietas enormes que dividen el modelo
    crack_seed = vertex_array[0] * 13.7 + vertex_array[2] * 29.3
    crack_pattern = abs(np.sin(crack_seed * 4.0))  # Frecuencia de grietas más amplia
    
    if crack_pattern > 0.6:  # Grietas más frecuentes y profundas
        crack_depth = (crack_pattern - 0.6) / 0.4 * 0.15  # Grietas mucho más profundas
        vertex_array += normal_array * (-crack_depth)
        
        # Agregar ramificación de grietas
        branch_x = np.sin(crack_seed * 25.0) * crack_depth * 2.0
        branch_z = np.cos(crack_seed * 30.0) * crack_depth * 2.0
        vertex_array += np.array([branch_x, 0, branch_z])
    
    # === DESCOMPOSICIÓN PROGRESIVA BASADA EN TIEMPO ===
    
    # El modelo se vuelve más destruido con el tiempo
    time_decay = np.sin(current_time * 0.3) * 0.1
    decay_pattern = np.sin(vertex_array[0] * 5.0 + current_time * 2.0) * np.cos(vertex_array[1] * 4.0 + current_time * 1.5)
    
    if abs(decay_pattern) > 0.3:
        progressive_decay = abs(decay_pattern) * time_decay * 0.5
        vertex_array += normal_array * (-progressive_decay)
    
    # === FALLA ESTRUCTURAL CATASTRÓFICA ===
    
    # Simular partes del modelo colapsando completamente
    failure_seed = vertex_array[0] * 67.3 + vertex_array[1] * 79.7 + vertex_array[2] * 89.1
    failure_pattern = np.sin(failure_seed * 3.0 + current_time * 0.5)
    
    if failure_pattern < -0.7:  # Áreas de colapso estructural
        collapse_intensity = (failure_pattern + 0.7) / 0.3
        # Colapso masivo hacia adentro
        vertex_array += normal_array * collapse_intensity * 0.6
        
        # Agregar desplazamiento caótico
        chaos_x = np.sin(failure_seed * 40.0) * abs(collapse_intensity) * 0.3
        chaos_y = np.cos(failure_seed * 45.0) * abs(collapse_intensity) * 0.3
        chaos_z = np.sin(failure_seed * 50.0) * abs(collapse_intensity) * 0.3
        vertex_array += np.array([chaos_x, chaos_y, chaos_z])
    
    # === PIPELINE DE TRANSFORMACIÓN ESTÁNDAR ===
    
    vt = [vertex_array[0], vertex_array[1], vertex_array[2], 1]
    nt = [normal_array[0], normal_array[1], normal_array[2], 0]
    
    vt = viewportMatrix @ projectionMatrix @ viewMatrix @ modelMatrix @ vt
    vt = vt.tolist()[0]
    
    # Transformar normal
    nt = np.linalg.inv(modelMatrix.T) @ nt
    nt = nt.tolist()[0]
    
    vt = [vt[0] / vt[3], vt[1] / vt[3], vt[2] / vt[3]]
    nt = [nt[0], nt[1], nt[2]]
    
    nt = nt / np.linalg.norm(nt)
    nt = nt.tolist()
    
    return vt, nt

def rusty_shader(**kwargs):
    # Shader de fragmento de metal oxidado con colores realistas de óxido y patrones
    u, v, w = kwargs["baryCoords"]
    A, B, C = kwargs["vertices"]
    current_time = kwargs.get("time", 0.0)
    
    # Calcular posición del mundo
    posA = np.array(A[:3])
    posB = np.array(B[:3])
    posC = np.array(C[:3])
    world_pos = posA * u + posB * v + posC * w
    
    # === PALETA DE COLORES DE ÓXIDO ===
    
    # Colores realistas de óxido y metal
    clean_metal = np.array([0.7, 0.7, 0.8])        # Acero/hierro limpio
    light_rust = np.array([0.8, 0.5, 0.3])         # Óxido temprano (naranja)
    medium_rust = np.array([0.7, 0.3, 0.1])        # Óxido medio (marrón rojizo)
    heavy_rust = np.array([0.5, 0.2, 0.1])         # Óxido pesado (marrón oscuro)
    rust_stains = np.array([0.6, 0.25, 0.05])      # Manchas de óxido (marrón-naranja)
    oxidation = np.array([0.4, 0.15, 0.05])        # Oxidación profunda (muy oscuro)
    patina_green = np.array([0.3, 0.5, 0.4])       # Pátina verde (oxidación de cobre)
    
    # === GENERACIÓN DE PATRONES DE ÓXIDO ===
    
    # Patrón principal de distribución de óxido
    rust_seed = world_pos[0] * 23.7 + world_pos[1] * 31.3 + world_pos[2] * 47.1
    
    # Áreas grandes de óxido
    rust_pattern1 = np.sin(rust_seed * 4.0) * np.cos(rust_seed * 3.0)
    rust_pattern2 = np.cos(rust_seed * 6.0) * np.sin(rust_seed * 5.0)
    rust_pattern3 = np.sin(rust_seed * 8.0) * np.cos(rust_seed * 7.0)
    
    # Combinar patrones de óxido
    combined_rust = (rust_pattern1 + rust_pattern2 + rust_pattern3) / 3.0
    rust_intensity = (combined_rust + 1.0) * 0.5  # Normalizar a [0,1]
    
    # === CLASIFICACIÓN DE CONDICIÓN SUPERFICIAL ===
    
    base_color = clean_metal.copy()
    
    if rust_intensity < 0.15:
        # Áreas de metal limpio
        base_color = clean_metal
        
    elif rust_intensity < 0.35:
        # Óxido superficial ligero
        mix_factor = (rust_intensity - 0.15) / 0.2
        base_color = clean_metal * (1 - mix_factor) + light_rust * mix_factor
        
    elif rust_intensity < 0.55:
        # Desarrollo de óxido medio
        mix_factor = (rust_intensity - 0.35) / 0.2
        base_color = light_rust * (1 - mix_factor) + medium_rust * mix_factor
        
    elif rust_intensity < 0.75:
        # Áreas de óxido pesado
        mix_factor = (rust_intensity - 0.55) / 0.2
        base_color = medium_rust * (1 - mix_factor) + heavy_rust * mix_factor
        
    elif rust_intensity < 0.9:
        # Corrosión severa
        mix_factor = (rust_intensity - 0.75) / 0.15
        base_color = heavy_rust * (1 - mix_factor) + oxidation * mix_factor
        
    else:
        # Oxidación completa
        base_color = oxidation
    
    # === PATRONES DE MANCHAS DE ÓXIDO ===
    
    # Agregar manchas y rayas de óxido
    stain_seed = world_pos[0] * 67.3 + world_pos[1] * 89.7 + world_pos[2] * 103.1
    stain_pattern = np.sin(stain_seed * 12.0) * np.cos(stain_seed * 8.0)
    
    if abs(stain_pattern) > 0.6:
        stain_intensity = (abs(stain_pattern) - 0.6) / 0.4
        base_color = base_color * (1 - stain_intensity * 0.5) + rust_stains * stain_intensity * 0.5
    
    # === EFECTOS DE PÁTINA ===
    
    # Agregar pátina verde en ciertas áreas (como elementos de cobre/bronce)
    patina_seed = world_pos[0] * 43.7 + world_pos[2] * 61.3  # Excluir Y para rayas horizontales
    patina_pattern = np.sin(patina_seed * 6.0)
    
    if patina_pattern > 0.7:
        patina_intensity = (patina_pattern - 0.7) / 0.3 * 0.4
        base_color = base_color * (1 - patina_intensity) + patina_green * patina_intensity
    
    # === VARIACIONES DE TEXTURA SUPERFICIAL ===
    
    # Agregar variaciones de textura para diferentes tipos de óxido
    texture_seed = world_pos[0] * 127.3 + world_pos[1] * 149.7 + world_pos[2] * 167.1
    texture_variation = np.sin(texture_seed * 25.0) * 0.1
    
    # Aplicar variación de textura
    base_color += texture_variation * np.array([0.1, 0.05, 0.02])  # Variación ligeramente rojiza
    
    # === RAYAS DE DESGASTE ===
    
    # Rayas verticales de desgaste (daño por agua)
    streak_pattern = np.sin(world_pos[0] * 15.0) * 0.1 * (1.0 - abs(world_pos[1]))  # Más fuerte en la parte inferior
    if world_pos[1] < 0:  # Partes inferiores más desgastadas
        streak_intensity = abs(streak_pattern) * 2.0
        if streak_intensity > 0.3:
            base_color = base_color * 0.8 + heavy_rust * 0.2
    
    # === PROGRESIÓN DE ÓXIDO BASADA EN TIEMPO ===
    
    # Óxido extendiéndose lentamente con el tiempo
    time_factor = 0.5 + 0.5 * np.sin(current_time * 0.1)  # Cambio muy lento
    rust_spread = rust_intensity + time_factor * 0.1
    
    if rust_spread > 0.8:
        spread_intensity = (rust_spread - 0.8) / 0.3
        base_color = base_color * (1 - spread_intensity * 0.3) + medium_rust * spread_intensity * 0.3
    
    # === INTEGRACIÓN DE ILUMINACIÓN ===
    
    # Obtener información de iluminación
    dirLight = kwargs.get("dirLight", (0, 0, -1))
    light_dir = -np.array(dirLight) / np.linalg.norm(np.array(dirLight))
    
    # Interpolar normal
    nA = np.array(A[3:6])
    nB = np.array(B[3:6])
    nC = np.array(C[3:6])
    normal = nA * u + nB * v + nC * w
    normal = normal / np.linalg.norm(normal)
    
    # Las superficies oxidadas tienen menos reflectividad
    n_dot_l = max(0.3, np.dot(normal, light_dir))  # Menos reflectivo que el metal limpio
    base_color *= n_dot_l
    
    # Agregar brillo ambiental sutil de óxido
    ambient_rust = 0.1 + 0.05 * rust_intensity
    base_color += np.array([0.05, 0.02, 0.01]) * ambient_rust
    
    # === FINALIZACIÓN ===
    
    # Asegurar que los colores estén en rango válido
    r, g, b = base_color
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    return [r, g, b]


