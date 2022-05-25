from abc import abstractmethod

from keras.activations import tanh
from keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Dense, InputSpec, Input, Reshape
from keras.models import Model
from keras.initializers.initializers_v2 import Zeros, RandomNormal, RandomUniform
from keras.regularizers import L1, L2


import learning_const
from learning_const import ShapeParamDimensionSize, AlbedoParamDimensionSize, BATCH_SIZE, RANDOM_SEED


class Decoder(Model):
    @property
    @abstractmethod
    def InputLayerUnits(self) -> int:
        """
        :return: Число нейронов входного слоя
        """
        pass

    @property
    @abstractmethod
    def LayerNamePrefix(self) -> str:
        '''
        :return: Префикс для названий слоёв
        '''
        pass

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self._init_set_name(self.LayerNamePrefix)
        self.model = self.ConfigureModel()
        print("Расшифрощик успешно создан.\n......................................")

        self.inputs = self.model.inputs
        self.outputs = self.model.outputs
        self.input_spec = InputSpec(shape=(BATCH_SIZE, self.InputLayerUnits))

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

        self.model.compile(optimizer, loss, metrics, loss_weights,
                           weighted_metrics, run_eagerly,
                           steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)


    def ConfigureModel(self) -> Model:
        # Описываем входные данные
        inputs = Input(shape=(self.InputLayerUnits,), batch_size=BATCH_SIZE, name=self.LayerNamePrefix + "_Input")

        # Входной слой у нас полносвязный, число нейронов зависит от того, что мы декодируем
        dense = Dense(13440, activation="relu",
                      name=self.LayerNamePrefix + "_FC",
                      kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02),
                      bias_initializer = Zeros(),
                         kernel_regularizer = L2())(inputs)

        # Подготавливаем данные к обратным свёрткам
        reshape = Reshape(target_shape=(learning_const.TEXTURE_SIZE[0] // 32, learning_const.TEXTURE_SIZE[1] // 32, 320),
                          name=self.LayerNamePrefix + "_ReshapeToConv")(dense)

        # Первая обратная свёртка (по обозначениям из статьи - FConv52).
        # Входные д-е 6х7х320, окно размером 3х3 с шагом 2, выходные д-е 12х14х160
        conv_52 = Conv2DTranspose(filters=160,
                                  kernel_size=(3, 3),
                                  strides=2,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv52",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(reshape)
        # После каждого свёрточного слоя (за исключением конечного) предписано
        # добавлять нормализацию и ReLU (на всякий решил поставить неумирающий ReLU, сеть-то большая)
        bn_52 = BatchNormalization(name=self.LayerNamePrefix + "_BN52", epsilon = 1.0e-05)(conv_52)
        relu_52 = LeakyReLU()(bn_52)

        # Вторая обратная свёртка (по обозначениям из статьи - FConv51).
        # Входные д-е 12х14х160, окно размером 3х3 с шагом 1, выходные д-е 12х14х256
        conv_51 = Conv2DTranspose(filters=256,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv51",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_52)
        bn_51 = BatchNormalization(name=self.LayerNamePrefix + "_BN51", epsilon = 1.0e-05)(conv_51)
        relu_51 = LeakyReLU()(bn_51)

        # Третья обратная свёртка (по обозначениям из статьи - FConv43).
        # Входные д-е 12х14х256, окно размером 3х3 с шагом 2, выходные д-е 24х28х256
        conv_43 = Conv2DTranspose(filters=256,
                                  kernel_size=(3, 3),
                                  strides=2,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv43",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_51)
        bn_43 = BatchNormalization(name=self.LayerNamePrefix + "_BN43", epsilon = 1.0e-05)(conv_43)
        relu_43 = LeakyReLU()(bn_43)

        # Четвёртая обратная свёртка (по обозначениям из статьи - FConv42).
        # Входные д-е 24х28х256, окно размером 3х3 с шагом 1, выходные д-е 24х28х128
        '''conv_42 = Conv2DTranspose(filters=128,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv42",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                                  bias_initializer = Zeros())(relu_43)
        bn_42 = BatchNormalization(name=self.LayerNamePrefix + "_BN42")(conv_42)
        relu_42 = LeakyReLU()(bn_42)'''

        # Пятая обратная свёртка (по обозначениям из статьи - FConv41).
        # Входные д-е 24х28х256, окно размером 3х3 с шагом 1, выходные д-е 24х28х128
        conv_41 = Conv2DTranspose(filters=128,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv41",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 2e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_43)
        bn_41 = BatchNormalization(name=self.LayerNamePrefix + "_BN41", epsilon = 1.0e-05)(conv_41)
        relu_41 = LeakyReLU()(bn_41)

        # Шестая обратная свёртка (по обозначениям из статьи - FConv33).
        # Входные д-е 24х28х128, окно размером 3х3 с шагом 2, выходные д-е 48х56х128
        conv_33 = Conv2DTranspose(filters=128,
                                  kernel_size=(3, 3),
                                  strides=2,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv33",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_41)
        bn_33 = BatchNormalization(name=self.LayerNamePrefix + "_BN33", epsilon = 1.0e-05)(conv_33)
        relu_33 = LeakyReLU()(bn_33)

        '''# Седьмая обратная свёртка (по обозначениям из статьи - FConv32).
        # Входные д-е 48х56х192, окно размером 3х3 с шагом 1, выходные д-е 48х56х96
        conv_32 = Conv2DTranspose(filters=96,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv32",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                                  bias_initializer = Zeros())(relu_33)
        bn_32 = BatchNormalization(name=self.LayerNamePrefix + "_BN32")(conv_32)
        relu_32 = LeakyReLU()(bn_32)'''

        # Восьмая обратная свёртка (по обозначениям из статьи - FConv31).
        # Входные д-е 48х56х128, окно размером 3х3 с шагом 1, выходные д-е 48х56х64
        conv_31 = Conv2DTranspose(filters=64,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv31",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_33)
        bn_31 = BatchNormalization(name=self.LayerNamePrefix + "_BN31", epsilon = 1.0e-05)(conv_31)
        relu_31 = LeakyReLU()(bn_31)

        # Девятая обратная свёртка (по обозначениям из статьи - FConv23).
        # Входные д-е 48х56х64, окно размером 3х3 с шагом 2, выходные д-е 96х112х64
        conv_23 = Conv2DTranspose(filters=64,
                                  kernel_size=(3, 3),
                                  strides=2,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv23",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_31)
        bn_23 = BatchNormalization(name=self.LayerNamePrefix + "_BN23", epsilon = 1.0e-05)(conv_23)
        relu_23 = LeakyReLU()(bn_23)

        '''# Десятая обратная свёртка (по обозначениям из статьи - FConv22).
        # Входные д-е 96х112х128, окно размером 3х3 с шагом 1, выходные д-е 96х112х64
        conv_22 = Conv2DTranspose(filters=64,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv22",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                                  bias_initializer = Zeros())(relu_23)
        bn_22 = BatchNormalization(name=self.LayerNamePrefix + "_BN22")(conv_22)
        relu_22 = LeakyReLU()(bn_22)

        # Одиннадцатая обратная свёртка (по обозначениям из статьи - FConv21).
        # Входные д-е 96х112х64, окно размером 3х3 с шагом 1, выходные д-е 96х112х64
        conv_21 = Conv2DTranspose(filters=64,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv21",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-03),
                                  bias_initializer = Zeros())(relu_22)
        bn_21 = BatchNormalization(name=self.LayerNamePrefix + "_BN21")(conv_21)
        relu_21 = LeakyReLU()(bn_21)'''

        # Двенадцатая обратная свёртка (по обозначениям из статьи - FConv13).
        # Входные д-е 96х112х64, окно размером 3х3 с шагом 2, выходные д-е 192х224х32
        conv_13 = Conv2DTranspose(filters=32,
                                  kernel_size=(3, 3),
                                  strides=2,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv13",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_23)
        bn_13 = BatchNormalization(name=self.LayerNamePrefix + "_BN13", epsilon = 1.0e-05)(conv_13)
        relu_13 = LeakyReLU()(bn_13)

        # Тринадцатая обратная свёртка (по обозначениям из статьи - FConv12).
        # Входные д-е 192х224х32, окно размером 3х3 с шагом 1, выходные д-е 192х224х12
        conv_12 = Conv2DTranspose(filters=12,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=None,
                                  name=self.LayerNamePrefix + "_Conv12",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_13)
        bn_12 = BatchNormalization(name=self.LayerNamePrefix + "_BN12", epsilon = 1.0e-05)(conv_12)
        relu_12 = LeakyReLU()(bn_12)

        # Тринадцатая обратная свёртка (по обозначениям из статьи - FConv11).
        # Входные д-е 192х224х32, окно размером 3х3 с шагом 1, выходные д-е 192х224х3
        # На выходе - активационная функция
        conv_11 = Conv2DTranspose(filters=3,
                                  kernel_size=(3, 3),
                                  strides=1,
                                  padding="same",
                                  activation=tanh,
                                  name=self.LayerNamePrefix + "_Conv11",
                                  kernel_initializer = RandomNormal(mean = 0.0, stddev = 1e-02, seed = RANDOM_SEED),
                                  bias_initializer = Zeros(),
                         kernel_regularizer = L2())(relu_12)

        return Model(inputs=inputs, outputs=conv_11)


    def ResetBatchNormLayers(self):
        for layer in self.model.layers:
            if layer.name.find("BN") > -1:
                print(f"Found BN layer with name {layer.name}")
                layer.trainable = False


class AlbedoDecoder(Decoder):
    """
    Класс сети для декодирования параметров отражения
    """

    @property
    def InputLayerUnits(self) -> int: return AlbedoParamDimensionSize

    @property
    def LayerNamePrefix(self) -> str: return "DA"

    def __init__(self):
        print("Конфигурируем расшифровщик альбедо...")

        super(AlbedoDecoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        #print("Получение текстуры альбедо из кодового вектора...")
        return super(AlbedoDecoder, self).call(inputs, training, mask)


class ShapeDecoder(Decoder):
    """
    Класс сети для декодирования параметров объёма
    """

    @property
    def InputLayerUnits(self) -> int: return ShapeParamDimensionSize

    @property
    def LayerNamePrefix(self) -> str: return "DS"

    def __init__(self):
        print("Конфигурируем расшифровщик формы...")

        super(ShapeDecoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        #print("Получение формы из кодового вектора...")
        return super(ShapeDecoder, self).call(inputs, training, mask)
