# Rasterizador 3D en Python

Este es un proyecto de un rasterizador 3D avanzado implementado en Python utilizando Pygame. Permite cargar modelos en formato `.obj`, aplicar transformaciones, cambiar entre diferentes shaders (incluyendo efectos especiales como oxidación y hologramas) y vistas de cámara.

## Características

- **Renderizado 3D completo**: Pipeline de rasterización con soporte para vértices, normales y texturas
- **Múltiples shaders**: Flat, Gouraud, Fragment y shaders especiales animados
- **Efectos especiales**: 
  - Shader de descomposición oxidada con destrucción extrema del modelo
  - Shader holográfico futurista con efectos de falla digital
  - Shader de visión térmica con paleta de temperatura y ondas de calor
  - Shader Tron/wireframe con efectos neón digitales y animación
- **Sistema de cámaras**: Múltiples ángulos de cámara predefinidos
- **Controles intuitivos**: Navegación y manipulación del modelo en tiempo real

## Cómo Usar

1.  **Requisitos:** Asegúrate de tener Python 3 y las dependencias instaladas.
    ```bash
    pip install pygame numpy
    ```
2.  **Ejecutar:** Navega al directorio `Rasterizer/General` y ejecuta el siguiente comando:
    ```bash
    python Rasterizer2025.py
    ```
3.  **Interactuar:** Una vez que la ventana se abra, puedes usar las siguientes teclas para interactuar con el programa.

## Controles

### Primitivas de Dibujo
- **1**: Cambiar a modo Puntos.
- **2**: Cambiar a modo Líneas.
- **3**: Cambiar a modo Triángulos (predeterminado).

### Vistas de Depuración
- **4**: Vista de Modelo.
- **5**: Vista de Cámara (View).
- **6**: Vista de Proyección.
- **7**: Vista de Viewport.
- **8**: Vista Completa (renderizado normal).

### Tomas de Cámara Predefinidas
- **Z**: Toma Media.
- **X**: Ángulo Bajo.
- **C**: Ángulo Alto.
- **V**: Ángulo Dutch.

### Shaders y Renderizado
- **I**: Aplicar Shader Plano (Flat).
- **O**: Aplicar Shader Gouraud.
- **P**: Aplicar Shader de Fragmentos (con texturas si están disponibles).
- **L**: Activar o desactivar la iluminación.
- **T**: Activar o desactivar las texturas (si el modelo tiene).
- **H**: Mostrar esta lista de controles en una ventana.

### Shaders Especiales
- **B**: Activar/Desactivar Shader de Descomposición Oxidada - Crea efectos extremos de oxidación, hoyos masivos, erosión catastrófica y destrucción estructural del modelo.
- **G**: Activar/Desactivar Shader Holográfico - Efectos futuristas con líneas de escaneo, fallas digitales y transparencia holográfica.
- **N**: Activar Shader de Visión Térmica - Simula visión térmica con paleta de colores de temperatura y ondas de calor.
- **J**: Activar Shader Tron/Wireframe - Efectos futuristas con wireframe neón, grid digital y corriente eléctrica animada.
- **M**: Desactivar todos los shaders especiales y volver al renderizado normal.

### Movimiento de la Cámara
- **Flecha Derecha**: Mover la cámara hacia la derecha.
- **Flecha Izquierda**: Mover la cámara hacia la izquierda.
- **Flecha Arriba**: Mover la cámara hacia arriba.
- **Flecha Abajo**: Mover la cámara hacia abajo.

### Manipulación del Modelo
- **D / A**: Rotar el modelo sobre su eje Z.
- **Q / E**: Rotar el modelo sobre su eje Y.
- **W / S**: Aumentar o disminuir la escala del modelo.

## Estructura del Proyecto

```
Rasterizer/
├── General/
│   ├── Rasterizer2025.py     # Aplicación principal
│   ├── gl.py                 # Motor de renderizado
│   ├── shaders.py            # Shaders de vértice y fragmento
│   ├── model.py              # Clase para manejo de modelos 3D
│   ├── texture.py            # Sistema de texturas
│   ├── MathLib.py            # Operaciones matemáticas 3D
│   ├── BMP_Writer.py         # Exportación de imágenes BMP
│   ├── show_controls.py      # Ventana de ayuda
│   ├── models/               # Modelos 3D (.obj)
│   └── textures/             # Texturas (.bmp)
└── FillPolygon/             # Implementación básica de rasterización
```

## Shaders Implementados

### Shaders Básicos
- **Flat Shader**: Iluminación uniforme por triángulo
- **Gouraud Shader**: Interpolación suave de normales
- **Fragment Shader**: Shader básico de fragmento
- **Textured Shader**: Renderizado con texturas

### Shaders Especiales
- **Rust Shader**: Simula oxidación extrema con:
  - Creación de hoyos masivos que perforan el modelo
  - Erosión superficial catastrófica
  - Astillamiento y desmoronamiento agresivo
  - Redes de grietas que dividen la geometría
  - Falla estructural con colapso progresivo
  - Efectos de tiempo que aumentan la destrucción

- **Hologram Shader**: Efectos holográficos futuristas con:
  - Líneas de escaneo animadas
  - Distorsión y fallas digitales
  - Transparencia dinámica
  - Ruido digital y corrupción de color
  - Efectos de parpadeo y pulsación

- **Thermal Vision Shader**: Simula visión térmica con:
  - Paleta de colores basada en temperatura (azul frío a rojo caliente)
  - Ondas de calor que distorsionan la geometría
  - Efectos de pulso térmico animado
  - Líneas de interferencia de escaneo térmico
  - Cálculo de temperatura basado en altura y ruido

- **Tron/Wireframe Shader**: Efectos futuristas digitales con:
  - Wireframe neón brillante en bordes
  - Grid digital animado en la superficie
  - Corriente eléctrica animada en las líneas
  - Ondas digitales y pulsos de escala
  - Efectos de escaneo vertical dinámico
  - Base oscura con acentos cian neón

## Características Técnicas

- **Pipeline 3D completo**: Transformaciones de modelo, vista, proyección y viewport
- **Iluminación**: Sistema de iluminación direccional con normales interpoladas
- **Texturas**: Soporte para mapeo de texturas con coordenadas UV
- **Animación**: Shaders animados basados en tiempo
- **Exportación**: Generación automática de archivos BMP al cerrar

## Modelos Incluidos

- `sword.obj` - Modelo principal de espada con textura
- `face.obj` - Modelo de cara
- `Container.obj` - Contenedor
- `baby.obj` - Modelo de bebé
- `model.obj` - Modelo genérico
