from MathLib import *


class Model(object):
    def __init__(self, name="Model"):
        self.name = name

        # Geometry
        self.vertices = []
        self.faces = []
        self.normals = []
        self.textureVertices = []
        self.faceTexCoords = []
        self.faceNormals = []

        # Transform
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.scale = [10, 10, 10]

        # Shaders & texture
        self.vertexShader = None
        self.fragmentShader = None
        self.texture = None

        # Materials (MTL) support
        self.materialLib = None  # mtl filename
        self.materials = {}  # name -> dict of props (e.g., {'map_Kd': '...'} )
        self.materialTextures = {}  # name -> Texture instance (loaded later)
        self.faceMaterials = []  # aligns with faces: material name used per face

    def GetModelMatrix(self):
        translateMat = TranslationMatrix(
            self.translation[0], self.translation[1], self.translation[2]
        )
        rotateMat = RotationMatrix(
            self.rotation[0], self.rotation[1], self.rotation[2]
        )
        scaleMat = ScaleMatrix(self.scale[0], self.scale[1], self.scale[2])
        return translateMat * rotateMat * scaleMat