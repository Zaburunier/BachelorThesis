import PIL.Image
import numpy.random
import tensorflow as tf

import win10toast
import learning_const
from data import NetworkDataGenerator
from network.NetworkFacade import NetworkFacade
from testing import tfg_render_test
from keras.preprocessing.image import array_to_img
from visualizer.ModelVisualizer import VisualizeFromNetworkData

#from numba import NumbaDeprecationWarning, NumbaWarning
import warnings
import multiprocessing


def main():
    #with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    #data_supplier = NetworkDataGenerator.NetworkDataGenerator(['AFLW2000'])#, 'AFLW2000', 'AFW', 'AFW_Flip'
    train_data, test_data = NetworkDataGenerator.CreateTrainTestDatasets(dataset_names=['AFLW2000'],
                                                                         single_batch_size=learning_const.BATCH_SIZE,
                                                                         validation_split=0.1, shuffled_split=True)

    #test_data.Render(0)

    #data_supplier.Render(8, True)
    facade = NetworkFacade()
    #facade.network.load_weights("C:\\Users\\lenya\\Downloads\\evaluation_code\\checkpoints\\")
    facade.TrainNetwork(train_data, test_data, shuffle_data = True, n_epochs = 50, pretrain = True)


def multiprocess_main():

    for i in range(80):
        p = multiprocessing.Process(target=train_with_multiprocessing)
        p.start()
        p.join()


def train_with_multiprocessing():
    #tf.config.run_functions_eagerly(True)
    train_data, test_data = NetworkDataGenerator.CreateTrainTestDatasets(dataset_names=['AFLW2000'],
                                                                         single_batch_size=learning_const.BATCH_SIZE,
                                                                         validation_split=0.1, shuffled_split=True)
    facade = NetworkFacade()

    #facade.network.load_weights("C:\\Users\\lenya\\Downloads\\evaluation_code\\checkpoints\\")
    facade.TrainNetworkOnSingleEpoch(train_data, test_data, shuffle_data = True, pretrain = True)


def test_main():
    data_supplier = NetworkDataGenerator.NetworkDataGenerator(['IBUG', 'IBUG_Flip'])#, 'AFLW2000', 'AFW', 'AFW_Flip'

    facade = NetworkFacade()
    shape_data, vertices, albedo_data, projection_data, converted_projection_data, \
    lightning_data, synthesized_images, synthesized_image_masks, \
    unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = facade.PassSingleImage(array_to_img(data_supplier.GetSingleImage(102)))


    VisualizeFromNetworkData(shape = vertices, projection = converted_projection_data, texture = unwarped_textures)

    #VisualizeFromNetworkData(shape = data_supplier.GetSingleFaceShape(0), projection = data_supplier.GetConvertedSingleProjection(0))#, texture = data_supplier.GetSingleTexture(0))

    shape_data, vertices, albedo_data, projection_data, converted_projection_data, \
    lightning_data, synthesized_images, synthesized_image_masks, \
    unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = facade.PassSingleImage(PIL.Image.open("My Images\\photo1.jpg"))
    VisualizeFromNetworkData(shape = vertices, projection = None, texture = unwarped_textures)


if __name__ == '__main__':
    numpy.random.seed(learning_const.RANDOM_SEED)
    tf.random.set_seed(learning_const.RANDOM_SEED)
    #tf.config.set_soft_device_placement(True)
    #tf.config.run_functions_eagerly(True)
    #tf.debugging.enable_check_numerics()
    #print(tf.config.list_logical_devices())
    #tf.compat.v1.disable_eager_execution()
    multiprocess_main()
    #main()
    #test_main()
