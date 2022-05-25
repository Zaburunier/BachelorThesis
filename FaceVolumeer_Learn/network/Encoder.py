from tensorflow import Tensor
from keras import Input
from keras.layers import Conv2D, AvgPool2D, LeakyReLU, BatchNormalization, Dense, Flatten, InputSpec, MaxPool2D
from keras.models import Model
from keras.initializers.initializers_v2 import Zeros, RandomNormal, RandomUniform
from keras.regularizers import L1, L2

from learning_const import ShapeParamDimensionSize, AlbedoParamDimensionSize, ProjectionParamDimensionSize, \
    LightParamDimensionSize, BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED


class Encoder(Model):

    def __init__(self, *args, **kwargs):
        '''
        Конструктор модели
        '''
        super(Encoder, self).__init__(*args, **kwargs)

        self.shape_encoder, \
        self.albedo_encoder, \
        self.projection_encoder, \
        self.light_encoder = self.ConfigureModel()
        print("Кодировщик успешно создан.\n......................................")

        self.inputs = self.shape_encoder.inputs
        self.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        self.outputs = [self.shape_encoder.outputs, self.albedo_encoder.outputs, self.projection_encoder.outputs,
                        self.light_encoder.outputs]

    def call(self, inputs, training=None, mask=None):
        return self.Encode(inputs, training, mask)

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        #super().compile(optimizer, loss, metrics, loss_weights,
        #                weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

        self.shape_encoder.compile(optimizer, loss, metrics, loss_weights,
                                   weighted_metrics, run_eagerly,
                                   steps_per_execution, **kwargs)

        self.albedo_encoder.compile(optimizer, loss, metrics, loss_weights,
                                    weighted_metrics, run_eagerly,
                                    steps_per_execution, **kwargs)

        self.projection_encoder.compile(optimizer, loss, metrics, loss_weights,
                                        weighted_metrics, run_eagerly,
                                        steps_per_execution, **kwargs)

        self.light_encoder.compile(optimizer, loss, metrics, loss_weights,
                                   weighted_metrics, run_eagerly,
                                   steps_per_execution, **kwargs)


    def ConfigureModel(self):
        """
        Собираем модель свёрточного кодировщика и возвращаем её
        :param loss_func: функция потерь, которая должна удовлетворять требованиям Tensorflow v2.
        :return: Экземпляр модели класса Sequential
        """
        print("Конфигурируем кодирующую сеть...")

        inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), batch_size=BATCH_SIZE, name="E_Input")
        # Нужен ли нам Rescale-слой?

        # rescaling = Rescaling(scale = 1.0 / 255, name = "E_Rescaling") (inputs)

        # Первая свёртка (по обозначениям из таблицы в статье - Conv11).
        # Входные д-е 224х224х3, окно размером 7х7 с шагом 2, выходные д-е 112х112х32
        conv_11 = Conv2D(filters=32,
                         kernel_size=(7, 7),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv11",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(inputs)

        # После каждого свёрточного слоя (за исключением конечного) предписано
        # добавлять нормализацию и ReLU (на всякий решил поставить неумирающий ReLU, сеть-то большая)
        bn_11 = BatchNormalization(name="E_BN11", epsilon = 1.0e-05)(conv_11)
        relu_11 = LeakyReLU()(bn_11)

        '''# Вторая свёртка (Conv12).
        # Входные д-е 112x112х32, окно размером 3х3 с шагом 1, выходные д-е 112x112х32
        conv_12 = Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv12",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_11)
        #bn_12 = BatchNormalization(name="E_BN12")(conv_12)
        relu_12 = LeakyReLU()(conv_12)'''

        # Третья свёртка (Conv21).
        # Входные д-е 112x112x32, окно размером 3х3 с шагом 1, выходные д-е 112x112х96
        conv_21 = Conv2D(filters=96,
                         kernel_size=(5, 5),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv21",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_11)

        bn_21 = BatchNormalization(name="E_BN21", epsilon = 1.0e-05)(conv_21)
        relu_21 = LeakyReLU()(bn_21)

        # Подвыборка
        # Входные д-е 112x112x96, окно размером 2х2 с шагом 1, выходные д-е 56x56х96
        #pool_21 = MaxPool2D(pool_size= (2, 2)) (relu_21)

        # Четвёртая свёртка (Conv22).
        # Входные д-е 112x112х64, окно размером 3х3 с шагом 1, выходные д-е 56х56х64
        conv_22 = Conv2D(filters=64,
                         kernel_size=(5, 5),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv22",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_21)

        bn_22 = BatchNormalization(name="E_BN22", epsilon = 1.0e-05)(conv_22)
        relu_22 = LeakyReLU()(bn_22)

        # Пятая свёртка (Conv23).
        # Входные д-е 56х56х64, окно размером 3х3 с шагом 1, выходные д-е 56х56х128
        conv_23 = Conv2D(filters=128,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv23",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_22)
        bn_23 = BatchNormalization(name="E_BN23", epsilon = 1.0e-05)(conv_23)
        relu_23 = LeakyReLU()(bn_23)

        # Шестая свёртка (Conv31).
        # Входные д-е 56х56х128, окно размером 3х3 с шагом 2, выходные д-е 28х28х192
        conv_31 = Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv31",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_23)
        bn_31 = BatchNormalization(name="E_BN31", epsilon = 1.0e-05)(conv_31)
        relu_31 = LeakyReLU()(bn_31)

        '''# Седьмая свёртка (Conv32).
        # Входные д-е 28х28х128, окно размером 3х3 с шагом 1, выходные д-е 28х28х96
        conv_32 = Conv2D(filters=96,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv32",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_31)
        bn_32 = BatchNormalization(name="E_BN32")(conv_32)
        relu_32 = LeakyReLU()(bn_32)'''

        # Подвыборка
        # Входные д-е 28х28х96(192!), окно размером 3х3 с шагом 1, выходные д-е 14х14х192
        #pool_32 = MaxPool2D(pool_size = (2, 2)) (relu_31)

        # Восьмая свёртка (Conv33).
        # Входные д-е 28x28x192, окно размером 3х3 с шагом 1, выходные д-е 14x14х256
        conv_33 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv33",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_31)
        bn_33 = BatchNormalization(name="E_BN33", epsilon = 1.0e-05)(conv_33)
        relu_33 = LeakyReLU()(bn_33)

        '''# Девятая свёртка (Conv41).
        # Входные д-е 28х28х192, окно размером 3х3 с шагом 2, выходные д-е 14х14х192
        conv_41 = Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv41",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_33)
        #bn_41 = BatchNormalization(name="E_BN41")(conv_41)
        relu_41 = LeakyReLU()(conv_41)'''

        # Десятая свёртка (Conv42).
        # Входные д-е 14х14х192, окно размером 3х3 с шагом 1, выходные д-е 14х14х128
        '''conv_42 = Conv2D(filters=128,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv42",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_41)
        bn_42 = BatchNormalization(name="E_BN42")(conv_42)
        relu_42 = LeakyReLU()(bn_42)'''

        '''# Одиннадцатая свёртка (Conv43).
        # Входные д-е 14х14х128(192!), окно размером 3х3 с шагом 1, выходные д-е 14х14х256
        conv_43 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv43",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_41)
        #bn_43 = BatchNormalization(name="E_BN43")(conv_43)
        relu_43 = LeakyReLU()(conv_43)'''

        # Двенадцатая свёртка (Conv51).
        # Входные д-е 14х14х256, окно размером 3х3 с шагом 2, выходные д-е 7х7х256
        conv_51 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         strides=2,
                         padding="same",
                         activation=None,
                         name="E_Conv51",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                         bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_33)
        bn_51 = BatchNormalization(name="E_BN51", epsilon = 1.0e-05)(conv_51)
        relu_51 = LeakyReLU()(bn_51)

        '''# Тринадцатая свёртка (Conv52).
        # Входные д-е 7х7х256, окно размером 3х3 с шагом 1, выходные д-е 7х7х160
        conv_52 = Conv2D(filters=160,
                         kernel_size=(3, 3),
                         strides=1,
                         padding="same",
                         activation=None,
                         name="E_Conv52",
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                         bias_initializer = Zeros())(relu_51)
        bn_52 = BatchNormalization(name="E_BN52")(conv_52)
        relu_52 = LeakyReLU()(bn_52)'''

        # Дальше пути сетей расходятся:
        # 1. Для формы и альбедо проводим только свёртку 53 и average pooling
        shape_conv_53 = Conv2D(filters=ShapeParamDimensionSize,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation="tanh",
                               name="E_ShapeConv53",
                               kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                               bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_51)

        albedo_conv_53 = Conv2D(filters=AlbedoParamDimensionSize,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same",
                                activation="tanh",
                                name="E_AlbedoConv53",
                                kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_51)

        # 2. Для параметров проекции и освещения общий слой свёртки
        projection_light_conv_53 = Conv2D(filters=64,
                                          kernel_size=(3, 3),
                                          strides=1,
                                          padding="same",
                                          activation = None,
                                          name="E_ProjLightConv53",
                                          kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                          bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_51)

        # Дальше в каждом случае идёт подвыборка
        shape_avg_pool = AvgPool2D(pool_size=(7, 7),
                                   strides=1,
                                   padding="valid")(shape_conv_53)
        albedo_avg_pool = AvgPool2D(pool_size=(7, 7),
                                    strides=1,
                                    padding="valid")(albedo_conv_53)
        projection_light_avg_pool = AvgPool2D(pool_size=(7, 7),
                                              strides=1,
                                              padding="valid")(projection_light_conv_53)

        # Для удобства превратим всё в одномерный формат
        shape_flatten = Flatten()(shape_avg_pool)
        albedo_flatten = Flatten()(albedo_avg_pool)
        projection_light_flatten = Flatten()(projection_light_avg_pool)

        # Для параметров проекции и освещения есть также выходной слой
        projection_output = Dense(ProjectionParamDimensionSize, name="E_ProjFC",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-05, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(), activation = "tanh", kernel_regularizer = L2()) (projection_light_flatten)
        light_output = Dense(LightParamDimensionSize, name="E_LightFC",
                             kernel_initializer = RandomNormal(mean = 0.0, stddev = 3e-02, seed = RANDOM_SEED),
                             bias_initializer = Zeros(), activation = "tanh", kernel_regularizer = L2())(projection_light_flatten)

        # Собираем модели
        shape_encoder = Model(inputs=inputs, outputs=shape_flatten, name="ShapeEncoder")
        shape_encoder.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        print("Кодировщик формы успешно создан.")

        albedo_encoder = Model(inputs=inputs, outputs=albedo_flatten, name="AlbedoEncoder")
        albedo_encoder.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        print("Кодировщик альбедо успешно создан.")

        projection_encoder = Model(inputs=inputs, outputs=projection_output, name="ProjectionEncoder")
        projection_encoder.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        print("Кодировщик ракурса успешно создан.")

        light_encoder = Model(inputs=inputs, outputs=light_output, name="LightEncoder")
        light_encoder.input_spec = InputSpec(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        print("Кодировщик освещения успешно создан.")

        return shape_encoder, albedo_encoder, projection_encoder, light_encoder


    def Encode(self, data: Tensor, is_training=True, mask = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        '''
        Кодирование входного изображения
        :param data: Исходные изображения
        :param is_training: Запускаемся ли мы в режиме обучения?
        :return: 4 кодирующих вектора (Shape, Albedo, Projection, Light соответственно)
        '''
        #print("Кодирование входных изображений...")
        return self.shape_encoder(data, is_training, mask), \
               self.albedo_encoder(data, is_training, mask), \
               self.projection_encoder(data, is_training, mask), \
               self.light_encoder(data, is_training, mask)


    def ResetBatchNormLayers(self):
        for layer in self.shape_encoder.layers:
            if layer.name.find("BN") > -1:
                print(f"Found BN layer with name {layer.name}")
                layer.trainable = False

        for layer in self.albedo_encoder.layers:
            if layer.name.find("BN") > -1:
                print(f"Found BN layer with name {layer.name}")
                layer.trainable = False

        for layer in self.projection_encoder.layers:
            if layer.name.find("BN") > -1:
                print(f"Found BN layer with name {layer.name}")
                layer.trainable = False

        for layer in self.light_encoder.layers:
            if layer.name.find("BN") > -1:
                print(f"Found BN layer with name {layer.name}")
                layer.trainable = False

