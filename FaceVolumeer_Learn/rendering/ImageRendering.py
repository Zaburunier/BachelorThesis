import math

import numpy as np
import pyrender.constants
from keras.preprocessing.image import img_to_array, array_to_img
import trimesh
from numpy import float32 as npfloat32
from pyrender import Scene, Mesh, PerspectiveCamera, OffscreenRenderer, Viewer
from trimesh import Trimesh, visual, Scene as TrimeshScene
from tensorflow import function, custom_gradient, float32 as tffloat32, cast, greater, reshape, \
    reduce_mean, numpy_function, convert_to_tensor, py_function, expand_dims, print as tfprint, math as tfmath, TensorSpec
from transformations import compose_matrix
import win10toast

import learning_const


@function(input_signature = (TensorSpec(dtype = tffloat32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3)),
                             TensorSpec(dtype = tffloat32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3)),
                             TensorSpec(dtype = tffloat32, shape = (learning_const.BATCH_SIZE, learning_const.MESH_VERTICES, 3))))
#@custom_gradient
def RenderImages(vertices, vertex_normals, vertex_colors):
    '''
    Генерация изображений из объёмных данных
    :param vertices: Положения вершин (размер [BATCH_SIZE, MESH_VERTICES, 3])
    :param vertex_normals: Нормали вершин (размер [BATCH_SIZE, MESH_VERTICES, 3])
    :param vertex_colors: Цвета вершин (размер [BATCH_SIZE, MESH_VERTICES, 3])
    :param projection_matrices: Положения моделей (размер [BATCH_SIZE, 4, 4])
    :param image_size: Размер выходного изображения
    :return: Набор созданных текстур (размер [BATCH_SIZE, image_size[0], image_size[1], 3])
    '''
    images = numpy_function(func = CreateRender,
                         inp = (vertices, vertex_normals, vertex_colors),
                         Tout = [tffloat32, tffloat32])
    #images = map_fn(fn = lambda input : numpy_function(func = PerformImageRender, inp = input, Tout = tffloat32), elems = (vertices, vertex_normals, vertex_colors, projection_matrices), fn_output_signature = (TensorSpec(points = (image_size[0], image_size[1], 3)), TensorSpec(points = (image_size[0], image_size[1]))))
    #print(images)
    colors, depths = images
    depths = expand_dims(cast(greater(depths, 0), dtype = tffloat32), axis = -1)

    #def grad_fn(upstream_colors, upstream_depths):
    #    #print(f"color grads = {upstream_colors};\ndepth grads = {upstream_depths}")
    #    mean_grad = reshape(reduce_mean(upstream_colors, axis = [1, 2, 3]), points = (const.BATCH_SIZE, 1 , 1))
    #    #tfprint(f"Mean grad of custom gradient: {mean_grad}")
    #    print(f"Mean grad of custom gradient: {mean_grad};\ninput points: {upstream_colors.points}; output_shape (vertices): {vertices.points}")
    #    if tfmath.is_nan(mean_grad):
    #        print(f"MEAN GRAD NAN CATCHED! Checking nan for input gradient: {tfmath.reduce_any(tfmath.is_nan(upstream_colors))}")
    #    return mean_grad * vertices, mean_grad * vertex_normals, mean_grad * vertex_colors, mean_grad * projection_matrices

    return (reshape(colors, shape = (learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)), reshape(depths, shape = (learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 1)))#, grad_fn


    # Внутренняя функция получения изображения (для экземпляра бэтча)
def CreateRender(vertices, normals, colors):
    '''

    :param vertices:
    :param normals:
    :param colors:
    :param view_matrix:
    :return:
    '''
    batch_size = vertices.shape[0]
    image_colors = np.ndarray(shape = [batch_size, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3], dtype = npfloat32)
    image_depths = np.ndarray(shape = [batch_size, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1]], dtype = npfloat32)

    for i in range(batch_size):
        image_colors[i], image_depths[i] = PerformImageRender(vertices[i], colors[i], normals[i])

    return (image_colors, image_depths)


def PerformImageRender(vertices, colors, normals):
    image_size = learning_const.IMAGE_SIZE


    # Задаём начальное положение объекта на сцене, помогая нейронной сети
    #view_matrix[:3, 3] += np.array([0.0, 0.0, -50.0])

    render_scene = Scene()

    # Добавляем на неё меш и камеру
    render_model = Mesh.from_points(vertices, colors, normals)

    model_pose = compose_matrix(translate = [0.0, 0.0, 0.0],
                                angles = [0.0, 0.0, 0.0],
                                scale = [1.0, 1.0, 1.0])

    camera_pose = compose_matrix(translate = np.array([0.0, 0.0, 50.0]),
                                angles = [0.0, 0.0, 0.0],
                                scale = [1.0, 1.0, 1.0])
    #print(f"Nan weights indices: {np.argwhere(np.isnan(np.ndarray(view_matrix)))}")
    #print(view_matrix)
    #print(f"Vertices centroid: {render_model.centroid};\ncamera matrix:\n{camera_pose}")
    render_scene.add(render_model, name="model", pose = model_pose)
    '''try:
        render_scene.add(render_model, name="model")
    except np.linalg.LinAlgError:
        print("ПОЙМАНО ИСКЛЮЧЕНИЕ LINALG")
        toast = win10toast.ToastNotifier()
        toast.show_toast(title="NUMPY LINALG ERROR", msg="Выскочили nan-ы!", duration=1)
        render_scene.add(render_model, name="model", pose=camera_pose)'''

    camera = PerspectiveCamera(yfov=math.pi / 6.0)
    render_scene.add(camera, name="cam", pose=camera_pose)

    '''render_viewer = Viewer(scene=render_scene)
    render_viewer.activate()'''
    #render_viewer.save_gif("gif_test")
    # Рендерим экземпляр бэтча
    render_buffer = OffscreenRenderer(viewport_width=image_size[0],
                                      viewport_height=image_size[1],
                                      point_size=1.0)
    rasterized = render_buffer.render(render_scene, flags=pyrender.constants.RenderFlags.FLAT)

    image_colors = convert_to_tensor(rasterized[0], dtype=tffloat32)
    image_depths = convert_to_tensor(rasterized[1], dtype=tffloat32)
    return image_colors, image_depths


def RenderSingleImage(vertices, normals, uv, image, view_matrix):
    tex = visual.TextureVisuals(uv=uv, image=image)
    #scene = TrimeshScene(geometry=[trimesh_model])
    #scene.set_camera(angles=[0.0, 0.0, 0.0], center=[0.0, 0.0, 120])
    #scene.show()
    return PerformImageRender(vertices,
                              tex.to_color().vertex_colors,
                              normals)



#@function
def ConvertTextureToVertexColors(uv, textures):
    def ConvertBatch(textures):
        color_array = []
        for i in range(textures.shape[0]):
            visual = trimesh.visual.TextureVisuals(uv = uv, image = array_to_img(textures[i]))
            color_array.append(np.asarray(visual.to_color().vertex_colors))
        return np.asarray(color_array, dtype = np.float32) / 255.0

    result = numpy_function(func = ConvertBatch, inp = (textures, ), Tout = tffloat32)
    return result