import numpy as np
from math import pi, sin, cos, isclose



def TranslationMatrix(x, y, z):
	
	return np.matrix([[1, 0, 0, x],
					  [0, 1, 0, y],
					  [0, 0, 1, z],
					  [0, 0, 0, 1]])



def ScaleMatrix(x, y, z):
	
	return np.matrix([[x, 0, 0, 0],
					  [0, y, 0, 0],
					  [0, 0, z, 0],
					  [0, 0, 0, 1]])



def RotationMatrix(pitch, yaw, roll):
	
	# Convertir a radianes
	pitch *= pi/180
	yaw *= pi/180
	roll *= pi/180
	
	# Creamos la matriz de rotaci�n para cada eje.
	pitchMat = np.matrix([[1,0,0,0],
						  [0,cos(pitch),-sin(pitch),0],
						  [0,sin(pitch),cos(pitch),0],
						  [0,0,0,1]])
	
	yawMat = np.matrix([[cos(yaw),0,sin(yaw),0],
						[0,1,0,0],
						[-sin(yaw),0,cos(yaw),0],
						[0,0,0,1]])
	
	rollMat = np.matrix([[cos(roll),-sin(roll),0,0],
						 [sin(roll),cos(roll),0,0],
						 [0,0,1,0],
						 [0,0,0,1]])
	
	return pitchMat * yawMat * rollMat

def create_view_matrix(position, rotation):
	# Crea la matriz de vista para la cámara
	# La matriz de vista es la inversa de la matriz de transformación de la cámara
	translate_matrix = TranslationMatrix(-position[0], -position[1], -position[2])
	
	pitch = rotation[0] * pi/180
	yaw = rotation[1] * pi/180
	roll = rotation[2] * pi/180

	# La inversa de una matriz de rotación es su transpuesta.
	# Aplicamos las rotaciones en orden inverso y con ángulos negados.
	pitch_mat = np.matrix([[1,0,0,0],
						  [0,cos(-pitch),-sin(-pitch),0],
						  [0,sin(-pitch),cos(-pitch),0],
						  [0,0,0,1]])
	
	yaw_mat = np.matrix([[cos(-yaw),0,sin(-yaw),0],
						[0,1,0,0],
						[-sin(-yaw),0,cos(-yaw),0],
						[0,0,0,1]])
	
	roll_mat = np.matrix([[cos(-roll),-sin(-roll),0,0],
						 [sin(-roll),cos(-roll),0,0],
						 [0,0,1,0],
						 [0,0,0,1]])

	# La matriz de rotación combinada es el producto de las matrices individuales
	# aplicadas en orden inverso de la rotación original.
	rotate_matrix = roll_mat * yaw_mat * pitch_mat

	return rotate_matrix @ translate_matrix

def create_projection_matrix(fov, aspect_ratio, near, far):
    # Crea una matriz de proyección en perspectiva
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.matrix([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def create_viewport_matrix(x, y, width, height):
    return np.matrix([
        [width / 2, 0, 0, x + width / 2],
        [0, height / 2, 0, y + height / 2],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]
    ])