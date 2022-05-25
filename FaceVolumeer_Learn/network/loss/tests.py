import PIL.Image
import keras.preprocessing.image

import learning_const
from network.loss import ShapeLoss, ProjectionLoss, TextureLoss, ReconstructionLoss, \
    AlbedoSymmetryLoss, AlbedoConstancyLoss, ShapeSmoothnessLoss, LandmarkLoss
import tensorflow as tf
from network.Volumeer2 import Volumeer2


def main():
    zeros = tf.zeros(shape = (2, 192, 224, 3))
    zeros_ones_concat = tf.concat([tf.zeros(shape = (2, 192, 112, 3)), tf.ones(shape = (2, 192, 112, 3))], axis = 2)
    noisy_zeros = tf.random.normal(shape = zeros.shape, mean = 0.0, stddev = 1.0)

    print("ALBEDO SYMMETRY LOSS")
    albedo_symmetry_loss = AlbedoSymmetryLoss.AlbedoSymmetryLoss()
    print(f"Значение для тензора zeros: {albedo_symmetry_loss.__call__(zeros, albedo_symmetry_loss.FlipAlbedos(zeros))}")
    print(f"Значение для конкатенированного из zeros и ones тензора: {albedo_symmetry_loss.__call__(zeros_ones_concat, albedo_symmetry_loss.FlipAlbedos(zeros_ones_concat))}")

    print("SHAPE RECONSTRUCTION LOSS")
    shape_loss = ShapeLoss.ShapeLoss()
    generic_face = Volumeer2.Load3DMM(learning_const.BASE_MODEL_DATA_DIRECTORY)
    face_shape = tf.convert_to_tensor(generic_face.vertices, dtype = tf.float32)
    print(f"Для модели с самой собой: {shape_loss.call(face_shape, face_shape)}")
    print(f"Для модели и зашумлённой версии (станд. норм. распр.): {shape_loss.call(face_shape, face_shape + tf.random.normal(shape = face_shape.shape))}")

    print("LANDMARK LOSS")
    landmark_loss = LandmarkLoss.LandmarkLoss()
    print(f"Для модели с самой собой: {landmark_loss.call(landmark_loss.GatherLandmarks(tf.expand_dims(face_shape, axis = 0)), landmark_loss.GatherLandmarks(tf.expand_dims(face_shape, axis = 0)))}")
    print(f"Для модели и зашумлённой версии (станд. норм. распр.): {landmark_loss.call(landmark_loss.GatherLandmarks(tf.expand_dims(face_shape, axis = 0)), landmark_loss.GatherLandmarks(tf.expand_dims(face_shape + tf.random.normal(shape = face_shape.shape), axis = 0)))}")
    print(f"Проверка для модели и полных нулей: {landmark_loss.call(landmark_loss.GatherLandmarks(tf.expand_dims(face_shape, axis = 0)), landmark_loss.GatherLandmarks(tf.expand_dims(tf.zeros(shape = face_shape.shape), axis = 0)))}")

    print("IMAGE RECONSTRUCTION LOSS")
    image_reconstruction_loss = ReconstructionLoss.ReconstructionLoss()
    img1 = keras.preprocessing.image.img_to_array(PIL.Image.open("image00040.png")) / 255.0
    img2 = keras.preprocessing.image.img_to_array(PIL.Image.open("image00041.png")) / 255.0
    print(f"Для нулей и единиц: {image_reconstruction_loss.call(tf.ones_like(img1), tf.zeros_like(img1))}")
    print(f"Для изображения с самим собой: {image_reconstruction_loss.call(img1, img1)}")
    print(f"Для изображения с зашумлённой версией: {image_reconstruction_loss.call(img1, img1 + tf.random.normal(shape = img1.shape))}")
    print(f"Для двух разных изображений: {image_reconstruction_loss.call(img1, img2)}")
    print(f"Для изображения и нулей: {image_reconstruction_loss.call(img1, tf.zeros_like(img1))}")

    print("TEXTURE RECONSTRUCTION LOSS")
    texture_reconstruction_loss = TextureLoss.TextureLoss()
    print(f"Для нулей и единиц: {texture_reconstruction_loss.call(tf.ones_like(img1), tf.zeros_like(img1))}")
    print(f"Для изображения с самим собой: {texture_reconstruction_loss.call(img1, img1)}")
    print(f"Для изображения с зашумлённой версией: {texture_reconstruction_loss.call(img1, img1 + tf.random.normal(shape = img1.shape))}")
    print(f"Для двух разных изображений: {texture_reconstruction_loss.call(img1, img2)}")
    print(f"Для изображения и нулей: {texture_reconstruction_loss.call(img1, tf.zeros_like(img1))}")

    print("ALBEDO CONSTANCY LOSS")
    albedo_constancy_loss = AlbedoConstancyLoss.AlbedoConstancyLoss()
    print(f"Для нулей и единиц: {albedo_constancy_loss.CalculateLoss(tf.ones(shape = (2, 192, 224, 3)), tf.zeros(shape = (2, 192, 224, 3)))}")
    print(f"Для нулей и нулей: {albedo_constancy_loss.CalculateLoss(tf.ones(shape = (2, 192, 224, 3)), tf.ones(shape = (2, 192, 224, 3)))}")
    print(f"Для нулей и станд.норм.распр: {albedo_constancy_loss.CalculateLoss(tf.ones(shape = (2, 192, 224, 3)), tf.random.normal(shape = (2, 192, 224, 3)))}")
    unwarped_tex = keras.preprocessing.image.img_to_array(PIL.Image.open("..\\..\\testing\\unwarped.png")) / 255.0
    unwarped_albedo = keras.preprocessing.image.img_to_array(PIL.Image.open("..\\..\\testing\\unwarped_no_shading.png")) / 255.0
    print(f"Для сгенерированного тестового изображения: {albedo_constancy_loss.CalculateLoss(tf.reshape(unwarped_tex, (1, unwarped_tex.shape[0], unwarped_tex.shape[1], unwarped_tex.shape[2])), tf.reshape(unwarped_albedo, (1, unwarped_tex.shape[0], unwarped_tex.shape[1], unwarped_tex.shape[2])))}")

    print("SHAPE SMOOTHNESS LOSS")
    shape_smoothness_loss = ShapeSmoothnessLoss.ShapeSmoothnessLoss()
    print(f"Значение для тензора zeros: {shape_smoothness_loss.__call__(shape_smoothness_loss.CalculateNeighboursAverageShape(zeros), shape_smoothness_loss.TrimShape(zeros))}")
    print(f"Значение для конкатенированного из zeros и ones тензора: {shape_smoothness_loss.__call__(shape_smoothness_loss.CalculateNeighboursAverageShape(zeros_ones_concat), shape_smoothness_loss.TrimShape(zeros_ones_concat))}")
    print(f"Значение для тензора из образцов стандартного нормального распределения: {shape_smoothness_loss.__call__(shape_smoothness_loss.CalculateNeighboursAverageShape(noisy_zeros), shape_smoothness_loss.TrimShape(noisy_zeros))}")

    print("SHAPE SMOOTHNESS LOSS FOR ALBEDO")
    print(f"Для сгенерированного тестового изображения: {shape_smoothness_loss.call(shape_smoothness_loss.CalculateNeighboursAverageShape(tf.reshape(unwarped_albedo, (1, unwarped_tex.shape[0], unwarped_tex.shape[1], unwarped_tex.shape[2]))), shape_smoothness_loss.TrimShape(tf.reshape(unwarped_albedo, (1, unwarped_tex.shape[0], unwarped_tex.shape[1], unwarped_tex.shape[2]))))}")


if __name__=="__main__":
    main()