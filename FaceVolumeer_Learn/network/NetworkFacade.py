import os
import logging

from numba import cuda

from PIL.Image import Image
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizer_v2.adam import Adam
from keras.models import load_model, save_model
from keras.preprocessing.image import array_to_img, img_to_array

import learning_const
from network.VolumeerNetwork import VolumeerNetwork
from network.Volumeer2 import Volumeer2
from data.NetworkDataGenerator import NetworkDataGenerator
from rendering import RenderingTools
from tools.FitLogCallback import FitCallback

import tensorflow as tf
import time

DATASET_INPUT_TYPESPEC = (tf.TensorSpec(dtype=tf.float32, shape=(learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)),
                          tf.TensorSpec(dtype=tf.float32, shape=(learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 1)),
                          tf.TensorSpec(dtype=tf.float32, shape=(learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 1)))
DATASET_TARGET_TYPESPEC = (tf.TensorSpec(dtype=tf.float32, shape=(learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 3)),
                           tf.TensorSpec(dtype=tf.float32, shape=(8,)),
                           tf.TensorSpec(dtype=tf.float32, shape=(learning_const.MESH_VERTICES, 3)))


class NetworkFacade:
    '''
    Класс для упрощённого взаимодействия с классом нейросети
    '''
    def __init__(self):
        '''
        Конструктор экземпляра (загружаем модель сети)
        '''
        self.network = self.InitializeNetwork()
        self.optimizer = Adam(learning_rate = 1.0e-03, beta_1 = 0.5)#, epsilon = 1e-04, clipvalue = .04, clipnorm = 3.0)
        self.epoch_counter = tf.Variable(initial_value = 0, dtype = tf.int32, trainable = False)

        self.checkpoint = tf.train.Checkpoint(root = self.network, epoch_counter = self.epoch_counter)#, optimizer = self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = learning_const.NETWORK_MODEL_SAVE_DIRECTORY, max_to_keep = 12)
        print(f"Checking manager last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        print(f"Called checkpoint manager to restore: {self.checkpoint_manager.restore_or_initialize()}")
        self.network.encoder.trainable = True
        self.network.shape_decoder.trainable = True
        self.network.albedo_decoder.trainable = True

        self.logger = FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "training log.csv", append=True)
        self.checkpoint_callback = ModelCheckpoint(
            filepath="D:\\Study\\Thesis\\FaceVolumeer_Learn\\checkpoints\\weights-epoch{epoch:02d}-{total_loss:.2f}-{shape_reconstruction_loss:.2f}.hdf5",
            save_best_only=True, save_weights_only=True, save_freq=200, mode="min", monitor="total_loss", verbose=1)

        #self.epoch_counter.assign_add(1) # ОДНОРАЗОВАЯ ШТУКА


    def InitializeNetwork(self) -> Volumeer2:
        '''
        Создание модели
        :return: Экземпляр класса VolumeerNetwork
        '''
        loadedModel = NetworkFacade.TryLoad()

        return Volumeer2() if loadedModel is None else loadedModel


    def TrainNetwork(self, train_data_generator : NetworkDataGenerator, test_data_generator : NetworkDataGenerator = None,
                     n_epochs = 1, max_steps_per_epoch = 0, max_validation_steps = 0, shuffle_data = False, pretrain = True):
        '''
        Обучение модели с помощью генератора данных
        :param data_generator: Генератор данных
        :param n_epochs: Число эпох обучения
        :param max_steps_per_epoch: Ограничитель числа шагов в эпохе (если 0, то никаких ограничений)
        :param shuffle_data: Перемешиваем ли данные после каждой эпохи?
        '''
        if max_steps_per_epoch == 0:
            max_steps_per_epoch = train_data_generator.__len__()

        if max_validation_steps == 0 and test_data_generator is not None:
            max_validation_steps = test_data_generator.__len__()

        if self.network.optimizer is None:
            self.network.optimizer = self.optimizer

        if self.network._is_compiled == False:
            self.network.compile(optimizer = self.network.optimizer)
            print("Optimizer:", self.network.optimizer.get_config())

        '''if (test_data_generator is None):
            dataset = tf.data.Dataset.from_generator(train_data_generator.__iter__, output_signature=(DATASET_INPUT_TYPESPEC, DATASET_TARGET_TYPESPEC))

            train_dataset = dataset
            validation_dataset = None
        else:
            train_dataset = tf.data.Dataset.from_generator(train_data_generator.__iter__, output_signature=(DATASET_INPUT_TYPESPEC, DATASET_TARGET_TYPESPEC))
            validation_dataset = tf.data.Dataset.from_generator(test_data_generator.__iter__, output_signature=(DATASET_INPUT_TYPESPEC, DATASET_TARGET_TYPESPEC))'''

        #train_batched = train_dataset.repeat().batch(batch_size=const.BATCH_SIZE, drop_remainder=True).prefetch(2)
        #test_batched = validation_dataset.repeat().batch(batch_size=const.BATCH_SIZE, drop_remainder=True).prefetch(2) if validation_dataset is not None else None



        #print(f"Non-trainable weights: {self.network.non_trainable_weights};\nTrainable weights: {self.network.trainable_weights}")

        self.network.using_landmark_loss_flag = True
        self.network.using_reconstruction_loss_flag = not pretrain
        self.network.using_albedo_constancy_loss_flag = not pretrain

        self.network.build(tf.TensorShape((learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)))
        #self.network.load_weights("weights on crash.hdf5")

        for epoch in range(n_epochs):
            tf.keras.backend.clear_session()
            self.network.current_epoch = self.epoch_counter.read_value().numpy()
            self.network.weights_to_train = self.network.encoder.trainable_weights if self.epoch_counter.read_value().numpy() % 2 == 0 else self.network.trainable_weights
            self.network.reset_states()

            '''if (shuffle_data):
                train_dataset = train_dataset.shuffle(buffer_size=4 * learning_const.BATCH_SIZE,
                                                      reshuffle_each_iteration=True, seed=learning_const.RANDOM_SEED)
                validation_dataset = validation_dataset.shuffle(buffer_size=4 * learning_const.BATCH_SIZE,
                                                                reshuffle_each_iteration=True,
                                                                seed=learning_const.RANDOM_SEED) if validation_dataset is not None else None

            train_batched = train_dataset.batch(batch_size=learning_const.BATCH_SIZE, drop_remainder=True).prefetch(2)
            test_batched = validation_dataset.batch(batch_size=learning_const.BATCH_SIZE,
                                                             drop_remainder=True).prefetch(2) if validation_dataset is not None else None'''
            
            #print(f"Volumeer2 with sample weight on before epoch begins {self.network.trainable_weights[0][0, 0, 0, 0]}")
            #print(f"НАЧАТА ЭПОХА ОБУЧЕНИЯ #{self.epoch_counter.read_value().numpy() + 1}\n--------------------------------------")
            '''if pretrain:
                if self.epoch_counter.read_value().numpy() % 2 == 1:
                    self.network.shape_decoder.trainable = False
                    self.network.albedo_decoder.trainable = True
                    #self.network.using_landmark_loss_flag = True
                else:
                    self.network.shape_decoder.trainable = True
                    self.network.albedo_decoder.trainable = True
                    #self.network.using_landmark_loss_flag = False
            #self.network.ResetBatchNormLayers()
            self.network.compile(optimizer = self.optimizer)'''

            #print(f"Model trainable: {self.network.shape_decoder.trainable}; layer 0 trainable: {self.network.shape_decoder.layers[0].trainable}")
            '''history = self.network.fit(x=train_batched, verbose=2, batch_size=learning_const.BATCH_SIZE,
                                       epochs=self.epoch_counter.read_value().numpy() + 1,
                                       initial_epoch=self.epoch_counter.read_value().numpy(),
                                       validation_data=test_batched, steps_per_epoch=max_steps_per_epoch,
                                       validation_steps=max_validation_steps,
                                       callbacks=[logger, csv_logger, checkpoint_callback])'''
            #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
            history = self.network.fit(x=train_data_generator, verbose=2, batch_size=learning_const.BATCH_SIZE,
                                       epochs=self.epoch_counter.read_value().numpy() + 1,
                                       initial_epoch=self.epoch_counter.read_value().numpy(),
                                       validation_data=test_data_generator, steps_per_epoch=max_steps_per_epoch,
                                       shuffle = shuffle_data,
                                       validation_steps=max_validation_steps,
                                       workers=1, use_multiprocessing=False,
                                       callbacks=[self.logger, self.csv_logger, self.checkpoint_callback])

            #self.SaveNetwork(epoch)
            #print(f"Volumeer2 with sample weight on after epoch ended {self.network.trainable_weights[0][0, 0, 0, 0]}")
            self.epoch_counter.assign_add(1)
            self.checkpoint_manager.save()

            #device = cuda.get_current_device()
            #device.reset()
            #cuda.select_device(0)
            #cuda.close()


    def TrainNetworkOnSingleEpoch(self, train_data_generator : NetworkDataGenerator, test_data_generator : NetworkDataGenerator = None,
                                  max_steps_per_epoch = 0, max_validation_steps = 0, shuffle_data = False, pretrain = True):
        if max_steps_per_epoch == 0:
            max_steps_per_epoch = train_data_generator.__len__()

        if max_validation_steps == 0 and test_data_generator is not None:
            max_validation_steps = test_data_generator.__len__()

        '''if pretrain:
            if self.epoch_counter.read_value().numpy() % 2 == 1:
                self.network.shape_decoder.trainable = False
                self.network.albedo_decoder.trainable = True
                # self.network.using_landmark_loss_flag = True
            else:
                self.network.shape_decoder.trainable = True
                self.network.albedo_decoder.trainable = True
                # self.network.using_landmark_loss_flag = False'''

        if self.network._is_compiled == False:
            self.network.compile(optimizer = self.optimizer)
            print("Optimizer:", self.network.optimizer.get_config())

        self.network.using_landmark_loss_flag = True
        self.network.using_reconstruction_loss_flag = not pretrain
        self.network.using_albedo_constancy_loss_flag = not pretrain

        self.network.build(tf.TensorShape((learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3)))
        self.network.reset_states()
        self.network.current_epoch = self.epoch_counter.read_value().numpy()
        self.network.weights_to_train = self.network.encoder.trainable_weights if self.epoch_counter.read_value().numpy() % 2 == 0 else self.network.trainable_weights

        history = self.network.fit(x=train_data_generator, verbose=2, batch_size=learning_const.BATCH_SIZE,
                                   epochs=self.epoch_counter.read_value().numpy() + 1,
                                   initial_epoch=self.epoch_counter.read_value().numpy(),
                                   validation_data=test_data_generator, steps_per_epoch=max_steps_per_epoch,
                                   shuffle=shuffle_data,
                                   validation_steps=max_validation_steps,
                                   workers=1, use_multiprocessing=False,
                                   callbacks=[self.logger, self.csv_logger, self.checkpoint_callback])

        self.epoch_counter.assign_add(1)
        self.checkpoint_manager.save()
        time.sleep(0.5)
        return None





    def PassSingleImage(self, img : Image):
        '''
        Пропускание изображения через нейросеть с целью получения полного отчёта о работе
        :param img: Исходное изображение
        :return: Кортеж данных в следующем виде:
        1 - Объём лица, представленный в виде текстуры (данные из нейросети, размер [TEXTURE_SIZE[0], TEXTURE_SIZE[1], 3]);\n
        2 - Альбедо лица, представленное в виде текстуры (данные из нейросети, размер [TEXTURE_SIZE[0], TEXTURE_SIZE[1], 3]);\n
        3 - Ракурс лица, три первых числа - углы вращения камеры, три следующих - вектор от камеры до центра модели (данные из нейросети, размер [BATCH_SIZE, 6]);\n
        4 -
        '''
        self.GetInfoLogger().info("Подготовка данных...\n")
        image_batch = tf.tile(tf.expand_dims(img_to_array(img.resize((224, 224))) / 255.0, axis = 0), (learning_const.BATCH_SIZE, 1, 1, 1)) #tf.concat([tf.expand_dims(img_to_array(img.resize((224, 224))) / 255.0, axis = 0), tf.zeros((learning_const.BATCH_SIZE - 1, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 3))], axis = 0)

        network_data, render_data = self.network.predict_step((image_batch, tf.zeros((learning_const.BATCH_SIZE, learning_const.IMAGE_SIZE[0], learning_const.IMAGE_SIZE[1], 1)), tf.zeros((learning_const.BATCH_SIZE, learning_const.TEXTURE_SIZE[0], learning_const.TEXTURE_SIZE[1], 1))))
        shape_data, vertices, albedo_data, projection_data, converted_projection_data, lightning_data = network_data
        synthesized_images, synthesized_image_masks, unwarped_textures, unwarped_shadings, unwarped_normals, rotated_vertices = render_data
        self.GetInfoLogger().info("\nВозвращаем результаты обработки...")
        return shape_data[0], vertices[0], albedo_data[0], projection_data[0], converted_projection_data[0], lightning_data[0], synthesized_images[0], synthesized_image_masks[0], unwarped_textures[0], unwarped_shadings[0], unwarped_normals[0],  rotated_vertices[0]


    def GetInfoLogger(self) -> logging.Logger:
        return self.network.logger


    def SaveNetwork(self, epoch_num):
        '''
        Сохранение модели в виде файла
        '''
        save_model(model = self.network, filepath = learning_const.NETWORK_MODEL_SAVE_DIRECTORY + learning_const.NETWORK_MODEL_SAVE_NAME) # + "_" + str(epoch_num)



    @classmethod
    def TryLoad(cls):
        '''
        Пробуем найти сохранённую модель для продолжения обучения
        :return: Найденный экземпляр класса Volumeer2 или None при его отсутствии
        '''
        if os.path.exists(learning_const.NETWORK_MODEL_SAVE_DIRECTORY + learning_const.NETWORK_MODEL_SAVE_NAME):
            # Если мы видим в нужной директории файлы, то это означает, что мы продолжаем обучение, прерванное ранее
            print("В папке" + (learning_const.NETWORK_MODEL_SAVE_DIRECTORY) + "обнаружена сохранённая модель" +
                  learning_const.NETWORK_MODEL_SAVE_NAME + ".")
            return load_model(learning_const.NETWORK_MODEL_SAVE_DIRECTORY + learning_const.NETWORK_MODEL_SAVE_NAME)
        else:
            # Если файлов нет, то мы начинаем обучение сначала
            print("Сохранённые модели не обнаружены.")
            return None