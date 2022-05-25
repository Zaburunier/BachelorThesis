import numpy as np
import learning_const
from trimesh import visual, Scene, Trimesh
from network import Volumeer2
from data.base_model import LandmarkData
from rendering import RenderingTools
from keras.preprocessing.image import array_to_img, img_to_array
from rendering import ProjectionTools


def VisualizeFromNetworkData(shape, projection = None, texture = None):
    '''
    Метод визуализации образца с помощью данных, полученных из нейросетей
    :param shape: Форма лица (в готовом для визуализации виде) в формате numpy.array размерность [n_vertices, 3]
    :param projection: Ракурс лица (требуется статистическое преобразование, но преобразование к матрице произойдёт внутри) в формате numpy.array длиной 8
    :param texture: Текстура лица (преобразование в Image произойдёт внутри) в формате numpy.array произвольного размере
    :return: Окно просмотра
    '''
    vertices = shape
    if projection is not None:
        '''projection = np.reshape(projection, (4, 2)).T
        row1 = projection[0, 0:3] / np.linalg.norm(projection[0, 0:3])
        row2 = projection[1, 0:3] / np.linalg.norm(projection[1, 0:3])
        row3 = np.cross(row1, row2)
        row1 = np.concatenate([np.reshape(row1, (1, 3)), np.zeros((1, 1))], axis = 1)
        row2 = np.concatenate([np.reshape(row2, (1, 3)), np.zeros((1, 1))], axis = 1)
        row3 = np.concatenate([np.reshape(row3, (1, 3)), np.zeros((1, 1))], axis = 1)
        new_projection = np.concatenate([row1, row2, row3], axis = 0)

        # Применяем вращение
        vertices = np.concatenate([vertices, np.ones(shape = (vertices.shape[0], 1))], axis = 1)
        vertices = np.matmul(new_projection, vertices.T).T'''
        projection_matrices = ProjectionTools.CreateRotationMatrices(np.expand_dims(projection, axis = 0))[0]
        vertices = ProjectionTools.RotatePointsWithMatrices(points = np.expand_dims(vertices, axis = 0), projection_matrices = np.expand_dims(projection_matrices, axis = 0))[0]

    # Смотрим базовые точки
    landmarks = LandmarkData.LandmarkData("D:\\Study\\Thesis\\FaceVolumeer_Learn\\3DMM_definition\\3DMM_keypoints.dat")
    colors = RenderingTools.GetVertexCoordColors(vertices=vertices)
    colors[landmarks.data] = [0.0, 0.0, 0.0, 1.0]

    model = Volumeer2.Volumeer2.Load3DMM(learning_const.BASE_MODEL_DATA_DIRECTORY)
    model.vertices = vertices
    model.fix_normals()
    if (texture is not None):
        model.visual = visual.TextureVisuals(uv = model.visual.uv, image = array_to_img(texture))

    scene = Scene(geometry = [model])
    scene.set_camera(center = model.centroid + np.array([0.0, 0.0, 50.0]))
    scene.camera.look_at([model.centroid])
    scene.show(smooth = False)

    # Проверяем вращение через 2х4
    '''landmarked_vertices = vertices[np.reshape(landmarks.data, (68, ))]
    landmarked_vertices_tf_like = np.take(vertices, np.reshape(landmarks.data, (68, )), axis = 0)
    rotated_landmarked_vertices_2d = np.matmul(new_2d_projection, landmarked_vertices.T).T
    rotated_landmarked_vertices_3d = np.matmul(new_projection, landmarked_vertices.T).T
    print(rotated_landmarked_vertices_2d)
    print(rotated_landmarked_vertices_3d)'''