import random
import struct
import pygame
import numpy as np
from MathLib import create_view_matrix, create_projection_matrix, create_viewport_matrix

POINTS = 0
LINES = 1
TRIANGLES = 2

class Renderer(object):
	def __init__(self, screen):
		self.screen = screen
		_, _, self.width, self.height = self.screen.get_rect()

		self.glColor(1,1,1)
		self.glClearColor(0,0,0)

		self.glClear()

		self.primitiveType = TRIANGLES  # Tipo de primitiva por defecto

		self.models = []

		self.activeModelMatrix = None
		self.activeVertexShader = None
		self.activeFragmentShader = None
		self.activeTexture = None
		self.active_shader = "all"  # Vista activa (depuración)

		# Iluminación
		self.dirLight = (0,0,1)
		self.lighting_enabled = True
		self.ambient = 0.2  # Luz ambiente para suavizar sombras (0..1)

		# Culling (descartar caras traseras)
		self.backface_culling = True

		# Cámara
		self.cameraPosition = [0,0,0]
		self.cameraRotation = [0,0,0]

		# Proyección
		self.projectionMatrix = None
		self.viewMatrix = None
		self.viewportMatrix = None
		
		# Tiempo para shaders animados
		self.time = 0.0

		# Buffer Z
		self.zBuffer = [[float('inf') for y in range(self.height)] for x in range(self.width)]

		# Fondo (opcional)
		self.background = None
		self._background_surface = None
		self._background_framebuffer = None  # x-major [width][height][3] uint8

		# Placeholders de matrices
		self.viewportMatrix = None
		self.viewMatrix = None
		self.projectionMatrix = None

	# --- Background helpers -------------------------------------------------
	def glLoadBackground(self, path):
		"""Carga un BMP de fondo para dibujarlo al limpiar."""
		try:
			self.background = _BackgroundTexture(path)
			# Construir superficie de pygame y copia de framebuffer al tamaño de pantalla
			self._build_background_assets()
		except Exception as e:
			# Ignorar errores al cargar fondo
			self.background = None
			self._background_surface = None
			self._background_framebuffer = None

	def _build_background_assets(self):
		"""Crea una Surface y una copia de framebuffer para el fondo."""
		self._background_surface = None
		self._background_framebuffer = None
		if not self.background:
			return
		# Convertir píxeles del fondo a arreglo numpy
		bg_h = self.background.height
		bg_w = self.background.width
		pixels = np.array(self.background.pixels, dtype=np.float32)
		pixels_u8 = (np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
		# Pygame surfarray usa (w,h,3)
		pixels_u8_wh = np.swapaxes(pixels_u8, 0, 1)
		try:
			src_surface = pygame.surfarray.make_surface(pixels_u8_wh)
		except Exception:
			# Sin surface si falla
			src_surface = None
		# Escalar a ventana si es necesario
		if src_surface and (bg_w != self.width or bg_h != self.height):
			src_surface = pygame.transform.smoothscale(src_surface, (self.width, self.height))
		# Guardar surface
		self._background_surface = src_surface
		# Construir copia de framebuffer (x-major)
		if src_surface:
			arr = pygame.surfarray.array3d(src_surface)
		else:
			# Usar píxeles crudos y redimensionar con numpy si aplica
			if bg_w != self.width or bg_h != self.height:
				# Escalado vecino más cercano
				x_idx = (np.linspace(0, bg_w - 1, self.width)).astype(int)
				y_idx = (np.linspace(0, bg_h - 1, self.height)).astype(int)
				resized = pixels_u8_wh[x_idx][:, y_idx]
				arr = resized
			else:
				arr = pixels_u8_wh
		# Convertir a listas anidadas x-major
		self._background_framebuffer = arr.tolist()

	def glClearBackground(self):
		"""Limpia pantalla y, si hay fondo, lo pinta en la ventana."""
		self.glClear()
		if not self.background:
			return
		# Ruta rápida: blit y copiar framebuffer del fondo
		if self._background_surface is None or self._background_framebuffer is None:
			self._build_background_assets()
		if self._background_surface is not None:
			self.screen.blit(self._background_surface, (0, 0))
		if self._background_framebuffer is not None:
			# Copia profunda para no mutar el cache del fondo
			self.frameBuffer = [row[:] for row in self._background_framebuffer]
			# Resetear zBuffer a infinito para dibujar encima del fondo
			self.zBuffer = [[float('inf') for _ in range(self.height)] for _ in range(self.width)]


	def glClearColor(self, r, g, b):
		# 0 - 1
		r = min(1, max(0,r))
		g = min(1, max(0,g))
		b = min(1, max(0,b))

		self.clearColor = [r,g,b]


	def glColor(self, r, g, b):
		# 0 - 1
		r = min(1, max(0,r))
		g = min(1, max(0,g))
		b = min(1, max(0,b))

		self.currColor = [r,g,b]

	def glClear(self):
		color = [int(i * 255) for i in self.clearColor]
		self.screen.fill(color)

		self.frameBuffer = [[color for y in range(self.height)]
							for x in range(self.width)]
		
		self.zBuffer = [[float('inf') for y in range(self.height)] for x in range(self.width)]

	def updateTime(self, deltaTime):
		"""Actualiza el tiempo para shaders animados"""
		self.time += deltaTime


	def glPoint(self, x, y, color = None, z = 0):
		# Pygame empieza a renderizar desde la esquina
		# superior izquierda, hay que voltear la Y

		x = round(x)
		y = round(y)

		if (0 <= x < self.width) and (0 <= y < self.height):
			if z < self.zBuffer[x][y]:
				self.zBuffer[x][y] = z
				color = [int(i * 255) for i in (color or self.currColor) ]

				self.screen.set_at((x,self.height - 1 - y ), color)

				self.frameBuffer[x][y] = color


	def glLine(self, p0, p1, color = None):
		# Algoritmo de Lineas de Bresenham
		# y = mx + b

		x0 = p0[0]
		x1 = p1[0]
		y0 = p0[1]
		y1 = p1[1]

		# Si el punto 0 es igual que el punto 1, solamente dibujar un punto
		if x0 == x1 and y0 == y1:
			self.glPoint(x0, y0)
			return

		dy = abs(y1 - y0)
		dx = abs(x1 - x0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dy = abs(y1 - y0)
		dx = abs(x1 - x0)

		offset = 0
		limit = 0.75
		m = dy / dx
		y = y0

		for x in range(round(x0), round(x1) + 1):
			if steep:
				self.glPoint(y, x, color or self.currColor)
			else:
				self.glPoint(x, y, color or self.currColor)

			offset += m

			if offset >= limit:
				if y0 < y1:
					y += 1
				else:
					y -= 1

				limit += 1

	def barycentricCoords(self, A, B, C, P):
		# Se saca el área de los subtriángulos y del triángulo
		# mayor usando el Shoelace Theorem, una fórmula que permite
		# sacar el área de un polígono de cualquier cantidad de vértices.

		areaPCB = abs((P[0]*C[1] + C[0]*B[1] + B[0]*P[1]) - 
					(P[1]*C[0] + C[1]*B[0] + B[1]*P[0]))

		areaACP = abs((A[0]*C[1] + C[0]*P[1] + P[0]*A[1]) - 
					(A[1]*C[0] + C[1]*P[0] + P[1]*A[0]))

		areaABP = abs((A[0]*B[1] + B[0]*P[1] + P[0]*A[1]) - 
					(A[1]*B[0] + B[1]*P[0] + P[1]*A[0]))

		areaABC = abs((A[0]*B[1] + B[0]*C[1] + C[0]*A[1]) - 
					(A[1]*B[0] + B[1]*C[0] + C[1]*A[0]))

		# Si el área del triángulo es 0, evitar división por 0
		if areaABC == 0:
			return None

		# Coordenadas baricéntricas = área subtriángulo / área total
		u = areaPCB / areaABC
		v = areaACP / areaABC
		w = areaABP / areaABC

		# Válidas si cada una ∈ [0,1] y u+v+w ≈ 1
		if 0<=u<=1 and 0<=v<=1 and 0<=w<=1 and abs(1 - (u+v+w)) < 0.001:
			return (u, v, w)
		else:
			return None


	def glTriangle(self, A, B, C):
		A_pos = [A[0], A[1], A[2]]
		B_pos = [B[0], B[1], B[2]]
		C_pos = [C[0], C[1], C[2]]
		
		minX = round(max(0, min(A_pos[0], B_pos[0], C_pos[0])))
		maxX = round(min(self.width - 1, max(A_pos[0], B_pos[0], C_pos[0])))
		minY = round(max(0, min(A_pos[1], B_pos[1], C_pos[1])))
		maxY = round(min(self.height - 1, max(A_pos[1], B_pos[1], C_pos[1])))

		for x in range(minX, maxX + 1):
			for y in range(minY, maxY + 1):
				coords = self.barycentricCoords(A_pos, B_pos, C_pos, (x, y))

				if coords is not None:
					u, v, w = coords
					# Interpolar z (profundidad)
					z = u * A_pos[2] + v * B_pos[2] + w * C_pos[2]

					shader_params = {
						"vertices": (A, B, C),
						"baryCoords": (u, v, w),
						"pixelColor": self.currColor,
						"dirLight": self.dirLight,
						"texture": self.activeTexture,
						"lighting_enabled": self.lighting_enabled,
						"time": self.time,
						"ambient": self.ambient
					}

					# Mezclar parámetros de shader por modelo (si existen)
					if hasattr(self, 'active_model') and getattr(self.active_model, 'shaderParams', None):
						shader_params.update(self.active_model.shaderParams)
					
					color = self.activeFragmentShader(**shader_params)
					self.glPoint(x, y, color=color, z=z)


	def glRender(self):
		# Contadores de depuración
		self.debug_stats = {
			"models": len(self.models),
			"faces": 0,
			"triangles": 0,
			"vertices_transformed": 0,
			"triangles_drawn": 0,
			"triangles_culled": 0,
		}

		for model in self.models:
			# Remember active model for shader param merging
			self.active_model = model
			# Por cada modelo en la lista, configurar estado activo
			self.activeModelMatrix = model.GetModelMatrix()
			self.activeVertexShader = model.vertexShader
			self.activeFragmentShader = model.fragmentShader
			self.activeTexture = model.texture
			# Luz direccional por modelo (opcional)
			self.dirLight = getattr(model, 'dirLight', self.dirLight)
			# Precalcular matriz de normales una vez por modelo
			try:
				normalMatrix = np.linalg.inv(self.activeModelMatrix.T)
			except Exception:
				normalMatrix = None

			# Dibujar por-cara para enlazar la textura correcta del material
			step = getattr(model, 'face_stride', 1) or 1
			for i in range(0, len(model.faces), step):
				self.debug_stats["faces"] += 1
				# Por cada cara, se dibujan sus vértices
				face = model.faces[i]
				if len(face) == 3:
					# Seleccionar textura desde material si existe
					face_mat = None
					if hasattr(model, 'faceMaterials') and i < len(model.faceMaterials):
						face_mat = model.faceMaterials[i]
					face_tex = None
					if face_mat and hasattr(model, 'materialTextures'):
						face_tex = model.materialTextures.get(face_mat, None)
					self.activeTexture = face_tex or model.texture
					self.debug_stats["triangles"] += 1
					a_idx, b_idx, c_idx = face[0]-1, face[1]-1, face[2]-1
					a = model.vertices[a_idx]
					b = model.vertices[b_idx]
					c = model.vertices[c_idx]

					# Normales por cara si existen; si no, usar [0,0,1]
					if i < len(model.faceNormals) and model.faceNormals[i] and len(model.faceNormals[i]) >= 3 and \
						all(idx is not None for idx in model.faceNormals[i][:3]):
						na_idx, nb_idx, nc_idx = model.faceNormals[i][0]-1, model.faceNormals[i][1]-1, model.faceNormals[i][2]-1
						if 0 <= na_idx < len(model.normals) and 0 <= nb_idx < len(model.normals) and 0 <= nc_idx < len(model.normals):
							na = model.normals[na_idx]
							nb = model.normals[nb_idx]
							nc = model.normals[nc_idx]
						else:
							na = nb = nc = [0, 0, 1]
					else:
						# Usar normales por defecto si las normales de cara no existen
						na = nb = nc = [0, 0, 1]
					
					# UV por cara si existen (robusto a None/rangos)
					texA = texB = texC = [0.0, 0.0]
					if i < len(model.faceTexCoords) and model.faceTexCoords[i]:
						ft = model.faceTexCoords[i]
						def sample_uv(idx):
							if idx is None:
								return [0.0, 0.0]
							uv_i = idx - 1
							if 0 <= uv_i < len(model.textureVertices):
								return model.textureVertices[uv_i]
							return [0.0, 0.0]
						if len(ft) >= 3:
							texA = sample_uv(ft[0])
							texB = sample_uv(ft[1])
							texC = sample_uv(ft[2])
					
					# Transformar vértices
					a_transformed, na_transformed = self.activeVertexShader(a, normal=na, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					b_transformed, nb_transformed = self.activeVertexShader(b, normal=nb, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					c_transformed, nc_transformed = self.activeVertexShader(c, normal=nc, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					self.debug_stats["vertices_transformed"] += 3
					
					# Construir buffer pequeño para este triángulo y dibujarlo
					tri_buf = []
					tri_buf.extend(a_transformed); tri_buf.extend(na_transformed); tri_buf.extend(texA)
					tri_buf.extend(b_transformed); tri_buf.extend(nb_transformed); tri_buf.extend(texB)
					tri_buf.extend(c_transformed); tri_buf.extend(nc_transformed); tri_buf.extend(texC)
					self.glDrawPrimitives(tri_buf, 8)

				elif len(face) == 4:
					# Seleccionar textura desde material si existe
					face_mat = None
					if hasattr(model, 'faceMaterials') and i < len(model.faceMaterials):
						face_mat = model.faceMaterials[i]
					face_tex = None
					if face_mat and hasattr(model, 'materialTextures'):
						face_tex = model.materialTextures.get(face_mat, None)
					self.activeTexture = face_tex or model.texture
					# Si es un cuadrilátero, dibujar como dos triángulos: ABC y ACD
					self.debug_stats["triangles"] += 2
					a_idx, b_idx, c_idx, d_idx = face[0]-1, face[1]-1, face[2]-1, face[3]-1
					a = model.vertices[a_idx]
					b = model.vertices[b_idx]
					c = model.vertices[c_idx]
					d = model.vertices[d_idx]

					# Normales por cara si existen; si no, usar [0,0,1]
					if i < len(model.faceNormals) and model.faceNormals[i] and len(model.faceNormals[i]) >= 4 and \
						all(idx is not None for idx in model.faceNormals[i][:4]):
						na_idx, nb_idx, nc_idx, nd_idx = model.faceNormals[i][0]-1, model.faceNormals[i][1]-1, model.faceNormals[i][2]-1, model.faceNormals[i][3]-1
						if all(0 <= idx < len(model.normals) for idx in (na_idx, nb_idx, nc_idx, nd_idx)):
							na = model.normals[na_idx]
							nb = model.normals[nb_idx]
							nc = model.normals[nc_idx]
							nd = model.normals[nd_idx]
						else:
							na = nb = nc = nd = [0, 0, 1]
					else:
						# Usar normales por defecto si las normales de cara no existen
						na = nb = nc = nd = [0, 0, 1]
					
					# UV por cara si existen (robusto a None/rangos)
					texA = texB = texC = texD = [0.0, 0.0]
					if i < len(model.faceTexCoords) and model.faceTexCoords[i]:
						ft = model.faceTexCoords[i]
						def sample_uv(idx):
							if idx is None:
								return [0.0, 0.0]
							uv_i = idx - 1
							if 0 <= uv_i < len(model.textureVertices):
								return model.textureVertices[uv_i]
							return [0.0, 0.0]
						if len(ft) >= 4:
							texA = sample_uv(ft[0])
							texB = sample_uv(ft[1])
							texC = sample_uv(ft[2])
							texD = sample_uv(ft[3])
					
					# Transformar vértices
					a_transformed, na_transformed = self.activeVertexShader(a, normal=na, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					b_transformed, nb_transformed = self.activeVertexShader(b, normal=nb, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					c_transformed, nc_transformed = self.activeVertexShader(c, normal=nc, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					d_transformed, nd_transformed = self.activeVertexShader(d, normal=nd, modelMatrix = self.activeModelMatrix, normalMatrix=normalMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					self.debug_stats["vertices_transformed"] += 4
					
					# Primer triángulo: A B C
					tri1 = []
					tri1.extend(a_transformed); tri1.extend(na_transformed); tri1.extend(texA)
					tri1.extend(b_transformed); tri1.extend(nb_transformed); tri1.extend(texB)
					tri1.extend(c_transformed); tri1.extend(nc_transformed); tri1.extend(texC)
					self.glDrawPrimitives(tri1, 8)
					# Segundo triángulo: A C D
					tri2 = []
					tri2.extend(a_transformed); tri2.extend(na_transformed); tri2.extend(texA)
					tri2.extend(c_transformed); tri2.extend(nc_transformed); tri2.extend(texC)
					tri2.extend(d_transformed); tri2.extend(nd_transformed); tri2.extend(texD)
					self.glDrawPrimitives(tri2, 8)



	def glDrawPrimitives(self, buffer, vertexOffset):
		if self.primitiveType == POINTS:
			for i in range(0, len(buffer), vertexOffset):
				x = buffer[i]
				y = buffer[i + 1]
				self.glPoint(x, y)

		elif self.primitiveType == LINES:
			for i in range(0, len(buffer), vertexOffset * 3):
				for j in range(3):
					x0 = buffer[i + vertexOffset * j + 0]
					y0 = buffer[i + vertexOffset * j + 1]
					x1 = buffer[i + vertexOffset * ((j + 1) % 3) + 0]
					y1 = buffer[i + vertexOffset * ((j + 1) % 3) + 1]
					self.glLine((x0, y0), (x1, y1))

		elif self.primitiveType == TRIANGLES:
			for i in range(0, len(buffer), vertexOffset * 3):
				A = [buffer[i + j + vertexOffset * 0] for j in range(vertexOffset)]
				B = [buffer[i + j + vertexOffset * 1] for j in range(vertexOffset)]
				C = [buffer[i + j + vertexOffset * 2] for j in range(vertexOffset)]

				# Culling fuera de pantalla via bounding box
				minX = round(max(0, min(A[0], B[0], C[0])))
				maxX = round(min(self.width - 1, max(A[0], B[0], C[0])))
				minY = round(max(0, min(A[1], B[1], C[1])))
				maxY = round(min(self.height - 1, max(A[1], B[1], C[1])))
				if maxX < 0 or maxY < 0 or minX >= self.width or minY >= self.height or minX > maxX or minY > maxY:
					if hasattr(self, 'debug_stats'):
						self.debug_stats["triangles_culled"] += 1
					continue

				# Backface culling en espacio de pantalla (winding)
				if self.backface_culling:
					ax, ay = A[0], A[1]
					bx, by = B[0], B[1]
					cx, cy = C[0], C[1]
					cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
					if cross <= 0:
						if hasattr(self, 'debug_stats'):
							self.debug_stats["triangles_culled"] += 1
						continue

				self.glTriangle(A, B, C)
				# Contar triángulos dibujados (debug)
				if hasattr(self, 'debug_stats'):
					self.debug_stats["triangles_drawn"] += 1

	def glSetCamera(self, translate, rotate):
		self.cameraPosition = translate
		self.cameraRotation = rotate
		self.viewMatrix = create_view_matrix(self.cameraPosition, self.cameraRotation)

	def glSetProjection(self, fov, aspect_ratio, near, far):
		self.projectionMatrix = create_projection_matrix(fov, aspect_ratio, near, far)

	def glSetViewport(self, x, y, width, height):
		self.viewportMatrix = create_viewport_matrix(x, y, width, height)








# Lector interno de BMP para el fondo (simple, colores float)
class _BackgroundTexture:
	def __init__(self, path):
		self.path = path
		self.width = 0
		self.height = 0
		self.pixels = []  # filas de [r,g,b] en 0..1
		self._read()

	def _read(self):
		with open(self.path, "rb") as image:
			image.seek(10)
			data_offset = struct.unpack('<I', image.read(4))[0]
			image.seek(18)
			self.width = struct.unpack('<i', image.read(4))[0]
			height_raw = struct.unpack('<i', image.read(4))[0]

			top_down = height_raw < 0
			self.height = abs(height_raw)

			image.seek(data_offset)

			# Cada fila BMP se rellena a múltiplos de 4 bytes
			row_stride = ((self.width * 3 + 3) // 4) * 4

			rows = []
			for _ in range(self.height):
				row_bytes = image.read(row_stride)
				row = []
				for x in range(self.width):
					b = row_bytes[x * 3 + 0] / 255.0
					g = row_bytes[x * 3 + 1] / 255.0
					r = row_bytes[x * 3 + 2] / 255.0
					row.append([r, g, b])
				rows.append(row)

			# Normalizar para que pixels[0] sea la fila superior
			if not top_down:
				rows.reverse()

			self.pixels = rows

	def get_color(self, u, v):
		if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and self.width > 0 and self.height > 0:
			x = int(u * (self.width - 1))
			y = int(v * (self.height - 1))
			return self.pixels[y][x]
		return [0.0, 0.0, 0.0]










