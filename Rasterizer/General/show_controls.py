import tkinter as tk
from tkinter import messagebox

def show_controls():
    controls_text = """
Controles del Rasterizador:

Primitivas de Dibujo:
- 1: Puntos
- 2: Líneas
- 3: Triángulos

Shaders y Vistas:
- 4: Vista de Modelo
- 5: Vista de Cámara (View)
- 6: Vista de Proyección
- 7: Vista de Viewport
- 8: Vista Completa (Todos los shaders)

Tomas de Cámara:
- Z: Toma Media
- X: Ángulo Bajo
- C: Ángulo Alto
- V: Ángulo Dutch

Shaders de Fragmentos:
- I: Shader Plano (Flat)
- O: Shader Gouraud
- P: Shader de Fragmentos (Predeterminado)
- L: Activar/Desactivar Iluminación
- T: Activar/Desactivar Texturas

Shaders Especiales:
- B: Activar Rusty Decay shader (Metal oxidado y corroído)
- N: Disponible para nuevos shaders
- M: Desactivar shaders especiales
- G: Activar/Desactivar shader Hologram (Futurista)

Movimiento de Cámara:
- Flecha Derecha: Mover a la derecha
- Flecha Izquierda: Mover a la izquierda
- Flecha Arriba: Mover hacia arriba
- Flecha Abajo: Mover hacia abajo

Manipulación del Modelo:
- D: Rotar en eje Z (sentido horario)
- A: Rotar en eje Z (sentido anti-horario)
- Q: Rotar en eje Y (sentido horario)
- E: Rotar en eje Y (sentido anti-horario)
- W: Aumentar escala
- S: Disminuir escala
"""
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    messagebox.showinfo("Controles", controls_text)
    root.destroy()

if __name__ == "__main__":
    show_controls()
