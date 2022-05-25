import numpy as np
import tensorflow as tf


def ProcessLambertianLight(face_normals, face_colors, light_direction = None, light_intensity = 1.0):
    '''
    Модель освещения Ламберта
    :param face_normals: Нормали треугольников
    :param face_colors: Цвета треугольников
    :param light_direction: Направление света
    :param light_intensity: Интенсивность света
    :return: Обновлённые цвета треугольников
    '''
    if light_direction is None:
        light_direction = np.array([0.0, -1.0, 0.0], dtype = np.float32)

    light_direction = np.asarray(light_direction / np.sqrt(np.sum(np.power(light_direction, 2.0), axis = 0)), dtype = np.float32)
    length = face_normals.shape[0]
    normals = tf.cast(face_normals, dtype = float32)
    colors = tf.cast(face_colors, dtype = tf.float32)
    light_vectors = tf.tile([-light_direction], multiples = [length, 1])
    dot_results = tf.math.maximum(tf.reduce_sum(normals * light_vectors, axis = 1, keepdims = False), 0)
    dot_results = tf.stack([dot_results, dot_results, dot_results, tf.ones(shape = [length])], axis = 1)
    #lit_colors = light_intensity * colors * dot_results
    return ModifyColorsByLight(colors, dot_results, light_intensity)


def ProcessPhongLight(face_normals, face_colors, light_direction = None, viewer_direction = None, light_intensity = 1.0):
    '''
    Модель освещения Ламберта
    :param face_normals: Нормали треугольников
    :param face_colors: Цвета треугольников
    :param light_direction: Направление света
    :param light_intensity: Интенсивность света
    :return: Обновлённые цвета треугольников
    '''
    if light_direction is None:
        light_direction = np.array([0.0, -1.0, 0.0], dtype = np.float32)

    if viewer_direction is None:
        viewer_direction = np.array([0.0, 0.0, 1.0], dtype = np.float32)

    light_direction = np.asarray(light_direction / np.sqrt(np.sum(np.power(light_direction, 2.0), axis = 0)), dtype = np.float32)
    viewer_direction = np.asarray(viewer_direction / np.sqrt(np.sum(np.power(viewer_direction, 2.0), axis = 0)), dtype = np.float32)
    h_direction = -(light_direction + viewer_direction) / 2.0
    h_direction = np.asarray(h_direction / np.sqrt(np.sum(np.power(h_direction, 2.0), axis = 0)), dtype = np.float32)

    length = face_normals.shape[0]
    normals = tf.cast(face_normals, dtype = tf.float32)
    colors = tf.cast(face_colors, dtype = tf.float32)
    h_vectors = tf.tile([h_direction], multiples = [length, 1])
    dot_results = tf.math.maximum(tf.reduce_sum(normals * h_vectors, axis = 1, keepdims = False), 0)
    dot_results = tf.stack([dot_results, dot_results, dot_results, tf.ones_like(dot_results)], axis = 1)
    #lit_colors = light_intensity * colors * dot_results
    return ModifyColorsByLight(colors, dot_results, light_intensity)


def ModifyColorsByLight(source_colors, light_dot_results, light_intensity):
    '''
    Применение правила изменения цветов по результатам моделирования света
    :param source_colors:
    :param light_dot_results:
    :param light_intensity:
    :return:
    '''
    #return clip_by_value(source_colors + light_intensity * light_dot_results, clip_value_min = 0.0, clip_value_max = 1.0).numpy()
    return (light_intensity * light_dot_results * source_colors).numpy()