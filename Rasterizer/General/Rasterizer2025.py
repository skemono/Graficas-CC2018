import pygame
from gl import *
from BMP_Writer import GenerateBMP
from model import Model
from shaders import *
from shaders import texturedShader
import shaders as SH
from MathLib import *

import os
import math
from texture import Texture
from show_controls import show_controls

def loadObj(file):
	model = Model(name=file[:-4])
	current_file_path = os.path.abspath(__file__)
	current_dir = os.path.dirname(current_file_path)
	obj_path = os.path.join(current_dir, "models", file)
	mtl_dir = os.path.dirname(obj_path)

	# Parser MTL mínimo
	def parse_mtl(mtl_path):
		mats = {}
		current = None
		if not os.path.exists(mtl_path):
			return mats
		with open(mtl_path, 'r') as mf:
			for line in mf:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				if line.startswith('newmtl '):
					current = line.split(None, 1)[1].strip()
					mats[current] = {}
				elif line.startswith('map_Kd') and current is not None:
					# map_Kd puede contener espacios; tomar el resto de la línea
					_, tex_rel = line.split(None, 1)
					mats[current]['map_Kd'] = tex_rel.strip()
		return mats

	with open(obj_path, "r") as f:
		current_material = None
		for line in f:
			if line.startswith('v '):
				parts = line.split()
				model.vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
			elif line.startswith('vt '):
				parts = line.split()
				model.textureVertices.append([float(parts[1]), float(parts[2])])
			elif line.startswith('vn '):
				parts = line.split()
				model.normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
			elif line.startswith('mtllib '):
				mtl_name = line.split(None, 1)[1].strip()
				model.materialLib = mtl_name
				# Cargar materiales ahora
				mtl_path = os.path.join(mtl_dir, mtl_name)
				model.materials = parse_mtl(mtl_path)
				# Precargar texturas para los materiales
				for mname, props in model.materials.items():
					tex_rel = props.get('map_Kd')
					if tex_rel:
						# Resolver primero relativo al directorio del MTL
						cand1 = os.path.join(mtl_dir, tex_rel)
						# O buscar por nombre base en la carpeta textures
						cand2 = os.path.join(current_dir, 'textures', os.path.basename(tex_rel))
						tpath = cand1 if os.path.exists(cand1) else cand2
						if os.path.exists(tpath):
							model.materialTextures[mname] = Texture(tpath)
			elif line.startswith('usemtl'):
				current_material = line.split(None, 1)[1].strip()
			elif line.startswith('f '):
				parts = line.split()
				face = []
				texCoords = []
				normals = []
				
				for vertex_data in parts[1:]:
					indices = vertex_data.split('/')
					
					face.append(int(indices[0]))
					
					if len(indices) > 1 and indices[1]:
						texCoords.append(int(indices[1]))
					else:
						texCoords.append(None)
					
					if len(indices) > 2 and indices[2]:
						normals.append(int(indices[2]))
					else:
						normals.append(None)
				
				model.faces.append(face)
				# Always append lists (aligned with faces) even if elements are None
				model.faceTexCoords.append(texCoords)
				model.faceNormals.append(normals)
				model.faceMaterials.append(current_material)
	# Registro único sobre el estado del MTL para este OBJ
	if model.materialLib and model.materials:
		print(f"[MTL] {file}: loaded {len(model.materials)} materials, {len(model.materialTextures)} textures from '{model.materialLib}'")
	else:
		print(f"[MTL] {file}: no mtllib/materials found; using fallback texture if provided")

	# Resumen de uso de materiales para validar el mapeo por cara
	if model.faces:
		total_faces = len(model.faces)
		used_names = [mn for mn in model.faceMaterials if mn]
		with_mat = len(used_names)
		with_tex = sum(1 for mn in used_names if mn in model.materialTextures)
		missing_tex_mats = sorted({mn for mn in used_names if mn not in model.materialTextures})
		print(f"[MTL] {file}: faces={total_faces}, faces_with_usemtl={with_mat}, faces_with_texture={with_tex}")
		if missing_tex_mats:
			print(f"[MTL] {file}: materials without map_Kd or missing file: {', '.join(missing_tex_mats)}")
	
	return model



width = 1920
height = 1080

screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()

rend = Renderer(screen)
rend.glSetProjection(fov=60, aspect_ratio=width/height, near=0.1, far=1000)
rend.glSetViewport(0, 0, width, height)
# Luz direccional: ~45° abajo-izquierda, leve -Z para iluminar caras frontales
rend.dirLight = (-0.577, -0.577, -0.577)
rend.backface_culling = False  # let Z-buffer decide by default

# Renderizar con triángulos por defecto
rend.primitiveType = TRIANGLES

# Intentar cargar un fondo BMP desde textures/background.bmp (opcional)
bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures", "background.bmp")
if os.path.exists(bg_path):
	rend.glLoadBackground(bg_path)

def make_model(obj_name, texture_name=None):
	m = loadObj(obj_name)
	m.vertexShader = vertexShader
	# Preferir texturas del MTL si existen
	if m.materialTextures:
		m.texture = None  # se usarán texturas por cara
		m.fragmentShader = texturedShader
		if texture_name:
			print(f"[MTL] {obj_name}: MTL textures take priority; ignoring fallback '{texture_name}'")
	else:
		# Alternativa: una sola textura si se proporciona
		if texture_name:
			tpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures", texture_name)
			if os.path.exists(tpath):
				m.texture = Texture(tpath)
				m.fragmentShader = texturedShader
				print(f"[MTL] {obj_name}: using fallback texture '{texture_name}' (no MTL textures)")
			else:
				m.texture = None
				m.fragmentShader = gouraudShader
				print(f"[MTL] {obj_name}: no MTL textures and fallback '{texture_name}' missing; rendering untextured")
		else:
			m.texture = None
			m.fragmentShader = gouraudShader
			print(f"[MTL] {obj_name}: no MTL textures and no fallback provided; rendering untextured")
	return m

# Build scene models (4)
models = []

model_specs = [
	("horse.obj", "horse.bmp"),
	("misspiggy.obj", "misspiggy.bmp"),
	("piggy.obj", "piggy.bmp"),
	("practical.obj", "practical.bmp")
]

# 1: toonShader, 2: rimIridescentShader, 3: lavaNoiseShader, 4: wireframeOverlayShader
initial_shader_by_model = {
	"horse": 1,
	"misspiggy": 2,
	"piggy": 3,
	"practical": 4,
}

# Parámetros opcionales por modelo al iniciar (valores de ejemplo)
initial_shader_params_by_model = {
	# "horse": {"levels": 5, "edge_width": 0.05},
	# "misspiggy": {"rim_power": 2.2, "rim_strength": 0.9},
	# "practical": {"scale": 1.3, "contrast": 1.8},
}

def _resolve_numbered_shader(choice:int):
	return {
		1: toonShader,
		2: rimIridescentShader,
		3: lavaNoiseShader,
		4: wireframeOverlayShader,
	}.get(choice, None)

for (oname, tname) in model_specs:
	try:
		m = make_model(oname, tname)
		# Posiciones iniciales para evitar que se superpongan
		m.translation = [0, 0, -10]
		m.scale = [0.1, 0.1, 0.1]
		m.rotation = [0, 0, 0]
		models.append(m)
		rend.models.append(m)
	except Exception as e:
		pass

# Separarlos un poco en X para mayor visibilidad
for i, m in enumerate(models):
	if m.name == "horse":
		m.translation[0] = -38
		m.translation[1] = -37.0
		m.translation[2] = -90.0
		m.scale = [0.03, 0.03, 0.03]
		m.rotation = [80, 180, 215]
		m.dirLight = (-1.0, -0.2, -0.2)
	elif m.name == "misspiggy":
		m.translation[0] = 45
		m.translation[1] = -40.0
		m.translation[2] = -100.0
		m.scale = [0.045, 0.045, 0.045]
		m.rotation = [0, -90, 0]
		# Luz desde la izquierda hacia el modelo (rayos +X), ligeramente hacia abajo/adelante
		m.dirLight = (1.0, -0.2, -0.2)
	elif m.name == "piggy":
		m.translation[0] = 5
		m.translation[1] = -22.65
		m.translation[2] = -100.0
		m.scale = [1, 1, 1]
		m.rotation = [0, -45, 0]
		m.dirLight = (-1.0, -0.2, -0.2)
	elif m.name == "practical":
		m.translation[0] = -40.0
		m.translation[1] = -10.0
		m.translation[2] = -89.0
		m.scale = [0.02, 0.02, 0.02]
		m.rotation = [0, 45, 0]
		m.dirLight = (-1.0, -0.2, -0.2)

# Aplicar shaders iniciales por modelo (si se especifica)
for m in models:
	choice = initial_shader_by_model.get(m.name)
	fn = _resolve_numbered_shader(choice) if choice is not None else None
	if fn is not None:
		m.fragmentShader = fn
	params = initial_shader_params_by_model.get(m.name)
	if params:
		m.shaderParams = dict(params)

# Estado de selección
selected_idx = 0

def current_model():
	return models[selected_idx]

def print_transform(tag, m):
	return





isRunning = True
while isRunning:

	# Inicio de temporización de fotograma
	frame_start = pygame.time.get_ticks()
	deltaTime = clock.tick(60) / 1000.0
	
	# Actualizar tiempo para shaders animados
	rend.time += deltaTime
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			isRunning = False

		elif event.type == pygame.KEYDOWN:
			# Capturar pantalla actual
			if event.key == pygame.K_F12:
				def _save_screenshot():
					base_dir = os.path.dirname(os.path.abspath(__file__))
					snap_dir = os.path.join(base_dir, "screenshots")
					os.makedirs(snap_dir, exist_ok=True)
					from datetime import datetime
					stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
					path = os.path.join(snap_dir, f"shot_{stamp}.png")
					try:
						pygame.image.save(screen, path)
						print(f"[shot] saved {path}")
					except Exception:
						pass
				_save_screenshot()
				continue
			# Asignar shaders numerados por modelo con Shift + 1..4
			if (event.mod & pygame.KMOD_SHIFT):
				if event.key == pygame.K_1:  # Shader 1: Toon
					current_model().fragmentShader = toonShader
					continue
				elif event.key == pygame.K_2:  # Shader 2: Rim + Iridescent
					current_model().fragmentShader = rimIridescentShader
					continue
				elif event.key == pygame.K_3:  # Shader 3: Lava Noise
					current_model().fragmentShader = lavaNoiseShader
					continue
				elif event.key == pygame.K_4:  # Shader 4: Wireframe Overlay
					current_model().fragmentShader = wireframeOverlayShader
					continue
			# Alternar modo de render: Puntos <-> Triángulos (previa rápida vs render completo)
			if event.key == pygame.K_f:
				if rend.primitiveType != TRIANGLES:
					rend.primitiveType = TRIANGLES
				else:
					rend.primitiveType = POINTS

			# Alternar backface culling para diagnosticar problemas de caras/sentido de giro
			elif event.key == pygame.K_F4:
				rend.backface_culling = not rend.backface_culling

			# Seleccionar el modelo a controlar
			elif event.key == pygame.K_TAB:
				selected_idx = (selected_idx + 1) % len(models)
			elif event.key == pygame.K_1 and len(models) >= 1:
				selected_idx = 0
			elif event.key == pygame.K_2 and len(models) >= 2:
				selected_idx = 1
			elif event.key == pygame.K_3 and len(models) >= 3:
				selected_idx = 2

			# Reasignar a teclas de función para evitar conflictos con números
			elif event.key == pygame.K_F5:
				rend.active_shader = "model"
			elif event.key == pygame.K_F6:
				rend.active_shader = "view"
			elif event.key == pygame.K_F7:
				rend.active_shader = "projection"
			elif event.key == pygame.K_F8:
				rend.active_shader = "viewport"
			elif event.key == pygame.K_F9:
				rend.active_shader = "all"
			# Tomas de cámara
			elif event.key == pygame.K_z: # Toma media
				rend.cameraPosition = [0, 0, 0]
				rend.cameraRotation = [0, 0, 0]
			elif event.key == pygame.K_x: # Ángulo bajo
				m = current_model()
				rend.cameraPosition = [m.translation[0], m.translation[1] - 5.5, m.translation[2] + 5]
				rend.cameraRotation = [45, 0, 0]
			elif event.key == pygame.K_c: # Ángulo alto
				m = current_model()
				rend.cameraPosition = [m.translation[0], m.translation[1] + 5.5, m.translation[2] + 5]
				rend.cameraRotation = [-50, 0, 0]
			elif event.key == pygame.K_v: # Ángulo holandés
				rend.cameraPosition = [0, 0, 0]
				rend.cameraRotation = [0, 0, 30]
			
			elif event.key == pygame.K_i:
				current_model().fragmentShader = flatShader
			elif event.key == pygame.K_o:
				current_model().fragmentShader = gouraudShader
			elif event.key == pygame.K_p:
				current_model().fragmentShader = fragmentShader
			elif event.key == pygame.K_l:  
				rend.lighting_enabled = not rend.lighting_enabled
			elif event.key == pygame.K_h:
				pass
			elif event.key == pygame.K_t:  
				m = current_model()
				if m.texture:
					m.fragmentShader = texturedShader if m.fragmentShader != texturedShader else gouraudShader
			elif event.key == pygame.K_b:
				m = current_model()
				rv = getattr(SH, 'rusty_vertex_shader', None)
				rf = getattr(SH, 'rusty_shader', None)
				if rv and rf:
					if m.vertexShader != rv:
						m.vertexShader = rv
						m.fragmentShader = rf
					else:
						m.vertexShader = vertexShader
						m.fragmentShader = texturedShader if m.texture else gouraudShader
			elif event.key == pygame.K_n:
				m = current_model()
				tv = getattr(SH, 'thermal_vertex_shader', None)
				tf = getattr(SH, 'thermal_shader', None)
				if tv and tf:
					m.vertexShader = tv
					m.fragmentShader = tf
			elif event.key == pygame.K_j:
				m = current_model()
				vv = getattr(SH, 'tron_vertex_shader', None)
				vf = getattr(SH, 'tron_shader', None)
				if vv and vf:
					m.vertexShader = vv
					m.fragmentShader = vf
			elif event.key == pygame.K_m:
				m = current_model()
				m.vertexShader = vertexShader
				if m.texture:
					m.fragmentShader = texturedShader
				else:
					m.fragmentShader = gouraudShader
			elif event.key == pygame.K_g:
				m = current_model()
				hv = getattr(SH, 'hologram_vertex_shader', None)
				hf = getattr(SH, 'hologramShader', None)
				if hv and hf:
					if m.vertexShader != hv:
						m.vertexShader = hv
						m.fragmentShader = hf
					else:
						m.vertexShader = vertexShader
						m.fragmentShader = texturedShader if m.texture else gouraudShader
	

	keys = pygame.key.get_pressed()

	# Controles de transformación para el modelo seleccionado
	m = current_model()
	changed = False

	# Trasladar con flechas
	move_speed = 5.0
	if keys[pygame.K_RIGHT]:
		m.translation[0] += move_speed * deltaTime
		changed = True
	if keys[pygame.K_LEFT]:
		m.translation[0] -= move_speed * deltaTime
		changed = True
	if keys[pygame.K_UP]:
		m.translation[1] += move_speed * deltaTime
		changed = True
	if keys[pygame.K_DOWN]:
		m.translation[1] -= move_speed * deltaTime
		changed = True

	# Rotar (A/D -> Z, Q/E -> Y, Z/C -> X)
	rot_speed = 45.0
	if keys[pygame.K_d]:
		m.rotation[2] += rot_speed * deltaTime
		changed = True
	if keys[pygame.K_a]:
		m.rotation[2] -= rot_speed * deltaTime
		changed = True
	if keys[pygame.K_q]:
		m.rotation[1] += rot_speed * deltaTime
		changed = True
	if keys[pygame.K_e]:
		m.rotation[1] -= rot_speed * deltaTime
		changed = True
	if keys[pygame.K_z]:
		m.rotation[0] += rot_speed * deltaTime
		changed = True
	if keys[pygame.K_c]:
		m.rotation[0] -= rot_speed * deltaTime
		changed = True

	# Escalar (W/S en todos los ejes)
	scale_step = 0.5
	if keys[pygame.K_w]:
		m.scale = [i + scale_step * deltaTime for i in m.scale]
		changed = True
	if keys[pygame.K_s]:
		m.scale = [max(0.01, i - scale_step * deltaTime) for i in m.scale]
		changed = True

	# Mover cámara con IJKL
	cam_speed = 10.0
	if keys[pygame.K_l]:
		rend.cameraPosition[0] += cam_speed * deltaTime
	if keys[pygame.K_j]:
		rend.cameraPosition[0] -= cam_speed * deltaTime
	if keys[pygame.K_i]:
		rend.cameraPosition[1] += cam_speed * deltaTime
	if keys[pygame.K_k]:
		rend.cameraPosition[1] -= cam_speed * deltaTime

	if changed:
		pass

	# Limpiar y dibujar el fondo (si existe)
	rend.glClearBackground()

	# Renderizado principal del modelo
	rend.glSetCamera(rend.cameraPosition, rend.cameraRotation)
	render_start = pygame.time.get_ticks()
	rend.glRender()
	render_end = pygame.time.get_ticks()
	

	#########################################

	pygame.display.flip()

	# Final de temporización de fotograma (sin prints de depuración)
	frame_end = pygame.time.get_ticks()


GenerateBMP("output.bmp", width, height, 3, rend.frameBuffer)

pygame.quit()