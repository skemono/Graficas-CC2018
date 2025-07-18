import pygame
from gl import *
from BMP_Writer import GenerateBMP
from model import Model
from shaders import *
import os
import math

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
			elif line.startswith('f '):
				parts = line.split()
				face = [int(p.split('/')[0]) for p in parts[1:]]
				model.faces.append(face)
	return model



width = 256
height = 256

screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()

rend = Renderer(screen)

# triangle3 = [[510,70], [550, 160], [570,80] ]

triangleModel = loadObj("sword.obj")

triangleModel.vertexShader = vertexShader
print("vertices", len(triangleModel.vertices))
rend.models.append(triangleModel)
triangleModel.translation = [width / 2, height / 2, 0]
triangleModel.scale = [3, 3, 3]
triangleModel.rotation[0] -= 90

 




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



	keys = pygame.key.get_pressed()

	if keys[pygame.K_RIGHT]:
		triangleModel.translation[0] += 10 * deltaTime
	if keys[pygame.K_LEFT]:
		triangleModel.translation[0] -= 10 * deltaTime
	if keys[pygame.K_UP]:
		triangleModel.translation[1] += 10 * deltaTime
	if keys[pygame.K_DOWN]:
		triangleModel.translation[1] -= 10 * deltaTime


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

	rend.glRender()
	

	#########################################

	pygame.display.flip()


GenerateBMP("output.bmp", width, height, 3, rend.frameBuffer)

pygame.quit()