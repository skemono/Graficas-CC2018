import random
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

		self.primitiveType = TRIANGLES

		self.models = []

		self.activeModelMatrix = None
		self.activeVertexShader = None
		self.activeFragmentShader = None
		self.activeTexture = None
		self.active_shader = "all"

		# Iluminación
		self.dirLight = (0,0,1)
		self.lighting_enabled = True

		# Cámara
		self.cameraPosition = [0,0,0]
		self.cameraRotation = [0,0,0]

		# Proyección
		self.projectionMatrix = None
		self.viewMatrix = None
		self.viewportMatrix = None

		# Buffer Z
		self.zBuffer = [[-float('inf') for y in range(self.height)] for x in range(self.width)]


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
		
		self.zBuffer = [[-float('inf') for y in range(self.height)] for x in range(self.width)]


	def glPoint(self, x, y, color = None, z = 0):
		# Pygame empieza a renderizar desde la esquina
		# superior izquierda, hay que voltear la Y

		x = round(x)
		y = round(y)

		if (0 <= x < self.width) and (0 <= y < self.height):
			if z > self.zBuffer[x][y]:
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

		# Si el área del triángulo es 0, retornar nada para
		# prevenir división por 0.
		if areaABC == 0:
			return None

		# Determinar las coordenadas baricéntricas dividiendo el 
		# área de cada subtriángulo por el área del triángulo mayor.
		u = areaPCB / areaABC
		v = areaACP / areaABC
		w = areaABP / areaABC

		# Si cada coordenada está entre 0 a 1 y la suma de las tres
		# es igual a 1, entonces son válidas.
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
					# Interpolar valor z
					z = u * A_pos[2] + v * B_pos[2] + w * C_pos[2]

					shader_params = {
						"vertices": (A, B, C),
						"baryCoords": (u, v, w),
						"pixelColor": self.currColor,
						"dirLight": self.dirLight,
						"texture": self.activeTexture,
						"lighting_enabled": self.lighting_enabled
					}
					
					color = self.activeFragmentShader(**shader_params)
					self.glPoint(x, y, color=color, z=z)


	def glRender(self):
		
		for model in self.models:
			# Por cada modelo en la lista, los dibujo
			# Agarrar su matriz modelo y vertexshader
			self.activeModelMatrix = model.GetModelMatrix()
			self.activeVertexShader = model.vertexShader
			self.activeFragmentShader = model.fragmentShader
			self.activeTexture = model.texture

			# Aqui vamos a guardar todos los vertices y su info correspondiente
			vertexBuffer = []

			for i in range(0, len(model.faces)):
				# Por cada cara, se dibujan los vertices
				# de la cara. Cada cara es un trio de vertices
				# que se mandan a dibujar
				face = model.faces[i]
				if len(face) == 3:
					a_idx, b_idx, c_idx = face[0]-1, face[1]-1, face[2]-1
					a = model.vertices[a_idx]
					b = model.vertices[b_idx]
					c = model.vertices[c_idx]

					# Verificar si existen normales de cara para esta cara
					if i < len(model.faceNormals) and model.faceNormals[i] and len(model.faceNormals[i]) >= 3:
						na_idx, nb_idx, nc_idx = model.faceNormals[i][0]-1, model.faceNormals[i][1]-1, model.faceNormals[i][2]-1
						na = model.normals[na_idx]
						nb = model.normals[nb_idx]
						nc = model.normals[nc_idx]
					else:
						# Usar normales por defecto si las normales de cara no existen
						na = nb = nc = [0, 0, 1]
					
					# Obtener coordenadas de textura si están disponibles
					texA = texB = texC = [0, 0]
					if i < len(model.faceTexCoords) and model.faceTexCoords[i]:
						if model.faceTexCoords[i][0] is not None:
							texA = model.textureVertices[model.faceTexCoords[i][0]-1]
						if model.faceTexCoords[i][1] is not None:
							texB = model.textureVertices[model.faceTexCoords[i][1]-1]
						if model.faceTexCoords[i][2] is not None:
							texC = model.textureVertices[model.faceTexCoords[i][2]-1]
					
					# Transformar cada vértice individualmente
					a_transformed, na_transformed = self.activeVertexShader(a, normal=na, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					b_transformed, nb_transformed = self.activeVertexShader(b, normal=nb, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					c_transformed, nc_transformed = self.activeVertexShader(c, normal=nc, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					
					# Agregar vértices transformados al buffer con coordenadas de textura
					vertexBuffer.extend(a_transformed)
					vertexBuffer.extend(na_transformed)
					vertexBuffer.extend(texA)
					vertexBuffer.extend(b_transformed)
					vertexBuffer.extend(nb_transformed)
					vertexBuffer.extend(texB)
					vertexBuffer.extend(c_transformed)
					vertexBuffer.extend(nc_transformed)
					vertexBuffer.extend(texC)

				elif len(face) == 4:
					# Si es un cuadrilatero, se dibujan dos triangulos
					# con los vertices de la cara
					# A B C D -> A B C y A C D
					a_idx, b_idx, c_idx, d_idx = face[0]-1, face[1]-1, face[2]-1, face[3]-1
					a = model.vertices[a_idx]
					b = model.vertices[b_idx]
					c = model.vertices[c_idx]
					d = model.vertices[d_idx]

					# Verificar si existen normales de cara para esta cara
					if i < len(model.faceNormals) and model.faceNormals[i] and len(model.faceNormals[i]) >= 4:
						na_idx, nb_idx, nc_idx, nd_idx = model.faceNormals[i][0]-1, model.faceNormals[i][1]-1, model.faceNormals[i][2]-1, model.faceNormals[i][3]-1
						na = model.normals[na_idx]
						nb = model.normals[nb_idx]
						nc = model.normals[nc_idx]
						nd = model.normals[nd_idx]
					else:
						# Usar normales por defecto si las normales de cara no existen
						na = nb = nc = nd = [0, 0, 1]
					
					# Obtener coordenadas de textura si están disponibles
					texA = texB = texC = texD = [0, 0]
					if i < len(model.faceTexCoords) and model.faceTexCoords[i]:
						if len(model.faceTexCoords[i]) >= 4:
							if model.faceTexCoords[i][0] is not None:
								texA = model.textureVertices[model.faceTexCoords[i][0]-1]
							if model.faceTexCoords[i][1] is not None:
								texB = model.textureVertices[model.faceTexCoords[i][1]-1]
							if model.faceTexCoords[i][2] is not None:
								texC = model.textureVertices[model.faceTexCoords[i][2]-1]
							if model.faceTexCoords[i][3] is not None:
								texD = model.textureVertices[model.faceTexCoords[i][3]-1]
					
					# Transformar cada vértice individualmente
					a_transformed, na_transformed = self.activeVertexShader(a, normal=na, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					b_transformed, nb_transformed = self.activeVertexShader(b, normal=nb, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					c_transformed, nc_transformed = self.activeVertexShader(c, normal=nc, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					d_transformed, nd_transformed = self.activeVertexShader(d, normal=nd, modelMatrix = self.activeModelMatrix, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix, viewportMatrix = self.viewportMatrix)
					
					# Primer triángulo: A B C
					vertexBuffer.extend(a_transformed)
					vertexBuffer.extend(na_transformed)
					vertexBuffer.extend(texA)
					vertexBuffer.extend(b_transformed)
					vertexBuffer.extend(nb_transformed)
					vertexBuffer.extend(texB)
					vertexBuffer.extend(c_transformed)
					vertexBuffer.extend(nc_transformed)
					vertexBuffer.extend(texC)
					
					# Segundo triángulo: A C D
					vertexBuffer.extend(a_transformed)
					vertexBuffer.extend(na_transformed)
					vertexBuffer.extend(texA)
					vertexBuffer.extend(c_transformed)
					vertexBuffer.extend(nc_transformed)
					vertexBuffer.extend(texC)
					vertexBuffer.extend(d_transformed)
					vertexBuffer.extend(nd_transformed)
					vertexBuffer.extend(texD)

			self.glDrawPrimitives(vertexBuffer, 8)



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

				self.glTriangle(A, B, C)

	def glSetCamera(self, translate, rotate):
		self.cameraPosition = translate
		self.cameraRotation = rotate
		self.viewMatrix = create_view_matrix(self.cameraPosition, self.cameraRotation)

	def glSetProjection(self, fov, aspect_ratio, near, far):
		self.projectionMatrix = create_projection_matrix(fov, aspect_ratio, near, far)

	def glSetViewport(self, x, y, width, height):
		self.viewportMatrix = create_viewport_matrix(x, y, width, height)










