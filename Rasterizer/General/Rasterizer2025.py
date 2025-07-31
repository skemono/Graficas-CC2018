import pygame
from gl import *
from BMP_Writer import GenerateBMP
from model import Model
from shaders import *
from shaders import texturedShader
from MathLib import *

import os
import math
from texture import Texture
from show_controls import show_controls

def loadObj(file):
	model = Model()
	current_file_path = os.path.abspath(__file__)
	current_dir = os.path.dirname(current_file_path)
	obj_path = os.path.join(current_dir, "models", file)

	with open(obj_path, "r") as f:
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
				if any(tc is not None for tc in texCoords):
					model.faceTexCoords.append(texCoords)
				if any(n is not None for n in normals):
					model.faceNormals.append(normals)
	
	return model



width = 1920
height = 1080

screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()

rend = Renderer(screen)
rend.glSetProjection(fov=60, aspect_ratio=width/height, near=0.1, far=1000)
rend.glSetViewport(0, 0, width, height)
rend.dirLight = (-5, 0, 0.3)  

# triangle3 = [[510,70], [550, 160], [570,80] ]


triangleModel = loadObj("sword.obj")

texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures", "sword.bmp")
if os.path.exists(texture_path):
	triangleModel.texture = Texture(texture_path)
	triangleModel.fragmentShader = texturedShader
else:
	triangleModel.texture = None
	triangleModel.fragmentShader = gouraudShader

triangleModel.vertexShader = vertexShader
print("vertices", len(triangleModel.vertices))
print("faces", len(triangleModel.faces))
print("normals", len(triangleModel.normals))
print("texture vertices", len(triangleModel.textureVertices))
print("face normals", len(triangleModel.faceNormals))
print("face tex coords", len(triangleModel.faceTexCoords))
rend.models.append(triangleModel)
triangleModel.translation = [0, 0, -10]
triangleModel.scale = [.1, .1, .1]
triangleModel.rotation = [90,0,0]





isRunning = True
while isRunning:

	deltaTime = clock.tick(60) / 1000.0
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			isRunning = False

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_1:
				rend.primitiveType = POINTS

			elif event.key == pygame.K_2:
				rend.primitiveType = LINES

			elif event.key == pygame.K_3:
				rend.primitiveType = TRIANGLES

			elif event.key == pygame.K_4:
				rend.active_shader = "model"
			elif event.key == pygame.K_5:
				rend.active_shader = "view"
			elif event.key == pygame.K_6:
				rend.active_shader = "projection"
			elif event.key == pygame.K_7:
				rend.active_shader = "viewport"
			elif event.key == pygame.K_8:
				rend.active_shader = "all"
			# Tomas de cámara
			elif event.key == pygame.K_z: # Toma media
				rend.cameraPosition = [0, 0, 0]
				rend.cameraRotation = [0, 0, 0]
			elif event.key == pygame.K_x: # Ángulo bajo
				rend.cameraPosition = [triangleModel.translation[0], triangleModel.translation[1] - 5.5, triangleModel.translation[2] + 5]
				rend.cameraRotation = [45, 0, 0]
			elif event.key == pygame.K_c: # Ángulo alto
				rend.cameraPosition = [triangleModel.translation[0], triangleModel.translation[1] + 5.5, triangleModel.translation[2] + 5]
				rend.cameraRotation = [-50, 0, 0]
			elif event.key == pygame.K_v: # Ángulo holandés
				rend.cameraPosition = [0, 0, 0]
				rend.cameraRotation = [0, 0, 30]
			
			elif event.key == pygame.K_i:
				triangleModel.fragmentShader = flatShader
			elif event.key == pygame.K_o:
				triangleModel.fragmentShader = gouraudShader
			elif event.key == pygame.K_p:
				triangleModel.fragmentShader = fragmentShader
			elif event.key == pygame.K_l:  
				rend.lighting_enabled = not rend.lighting_enabled
				print(f"Lighting {'enabled' if rend.lighting_enabled else 'disabled'}")
			elif event.key == pygame.K_h:
				show_controls()
			elif event.key == pygame.K_t:  
				if triangleModel.texture:
					triangleModel.fragmentShader = texturedShader if triangleModel.fragmentShader != texturedShader else gouraudShader
					print(f"Texture {'enabled' if triangleModel.fragmentShader == texturedShader else 'disabled'}")
	

	keys = pygame.key.get_pressed()

	if keys[pygame.K_RIGHT]:
		rend.cameraPosition[0] += 10 * deltaTime
	if keys[pygame.K_LEFT]:
		rend.cameraPosition[0] -= 10 * deltaTime
	if keys[pygame.K_UP]:
		rend.cameraPosition[1] += 10 * deltaTime
	if keys[pygame.K_DOWN]:
		rend.cameraPosition[1] -= 10 * deltaTime

	if keys[pygame.K_d]:
		triangleModel.rotation[2] += 20 * deltaTime
	if keys[pygame.K_a]:
		triangleModel.rotation[2] -= 20 * deltaTime

	if keys[pygame.K_q]:
		triangleModel.rotation[1] += 20 * deltaTime
	if keys[pygame.K_e]:
		triangleModel.rotation[1] -= 20 * deltaTime

	if keys[pygame.K_w]:
		triangleModel.scale =  [(i + deltaTime) for i in triangleModel.scale]
	if keys[pygame.K_s]:
		triangleModel.scale = [(i - deltaTime) for i in triangleModel.scale ]










	rend.glClear()

	# Escribir lo que se va a dibujar aqui
	rend.glSetCamera(rend.cameraPosition, rend.cameraRotation)
	rend.glRender()
	

	#########################################

	pygame.display.flip()


GenerateBMP("output.bmp", width, height, 3, rend.frameBuffer)

pygame.quit()