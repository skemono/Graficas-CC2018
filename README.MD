# Rasterizador 3D en Python

Este es un proyecto de un rasterizador 3D simple implementado en Python utilizando Pygame. Permite cargar modelos en formato `.obj`, aplicar transformaciones, cambiar entre diferentes shaders y vistas de cámara.

## Cómo Usar

1.  **Requisitos:** Asegúrate de tener Python 3 y Pygame instalado.
    ```bash
    pip install pygame
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

### Movimiento de la Cámara
- **Flecha Derecha**: Mover la cámara hacia la derecha.
- **Flecha Izquierda**: Mover la cámara hacia la izquierda.
- **Flecha Arriba**: Mover la cámara hacia arriba.
- **Flecha Abajo**: Mover la cámara hacia abajo.

### Manipulación del Modelo
- **D / A**: Rotar el modelo sobre su eje Z.
- **Q / E**: Rotar el modelo sobre su eje Y.
- **W / S**: Aumentar o disminuir la escala del modelo.
