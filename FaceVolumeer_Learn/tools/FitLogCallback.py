from keras.backend import clear_session
from keras.callbacks import Callback
import gc


class FitCallback(Callback):
    '''def on_train_batch_begin(self, batch, logs=None):
        super(FitCallback, self).on_train_batch_begin(batch, logs)

        print(f"Поймано начало {batch}-го бэтча")
        tf.print(f"Поймано начало {batch}-го бэтча")'''

    def on_train_begin(self, logs=None):
        #print("Starting training; got log keys: {}".format(keys))
        print("\tНачинаем тренировать модель, переданные аргументы:\n{}".format(logs))

    def on_train_end(self, logs=None):
        #print("Stop training; got log keys: {}".format(keys))
        print("\tЗакончили тренировать модель, переданные аргументы:\n{}".format(logs))

    def on_epoch_begin(self, epoch, logs=None):
        #print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        print(f"\tНачинаем эпоху обучения #{epoch}, переданные аргументы:\n{logs}")

    def on_epoch_end(self, epoch, logs=None):
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        print(f"\tЗавершили эпоху обучения #{epoch}, переданные аргументы:\n{logs}")

        gc.collect()
        clear_session()

    def on_test_begin(self, logs=None):
        #print("Start testing; got log keys: {}".format(keys))
        print(f"\tНачинаем оценку модели, переданные аргументы:\n{logs}")

    def on_test_end(self, logs=None):
        #print("Stop testing; got log keys: {}".format(keys))
        print(f"\tЗавершили оценку модели, переданные аргументы:\n{logs}")

    def on_predict_begin(self, logs=None):
        #print("Start predicting; got log keys: {}".format(keys))
        print(f"\tНачинаем прогнозирование, переданные аргументы:\n{logs}")

    def on_predict_end(self, logs=None):
        #print("Stop predicting; got log keys: {}".format(keys))
        print(f"\tЗакончили прогнозирование, переданные аргументы:\n{logs}")

    def on_train_batch_begin(self, batch, logs=None):
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
        print(f"\t\t(ОБУЧЕНИЕ) Получили на вход бэтч #{batch}, переданные аргументы:\n{logs}")

    def on_train_batch_end(self, batch, logs=None):
        #print("...Training: end of batch {}; got log:\n{}".format(batch, logs))
        print(f"\t\t(ОБУЧЕНИЕ) Закончили с бэтчем #{batch}, переданные аргументы:\n{logs}")

    def on_test_batch_begin(self, batch, logs=None):
        #print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
        print(f"\t\t(ОЦЕНКА) Получили на вход бэтч #{batch}, переданные аргументы:\n{logs}")

    def on_test_batch_end(self, batch, logs=None):
        #print("...Evaluating: end of batch {}; got log:\n{}".format(batch, logs))
        print(f"\t\t(ОЦЕНКА) Закончили с бэтчем #{batch}, переданные аргументы:\n{logs}")

    def on_predict_batch_begin(self, batch, logs=None):
        #print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
        print(f"\t\t(ПРОГНОЗ) Получили на вход бэтч #{batch}, переданные аргументы:\n{logs}")

    def on_predict_batch_end(self, batch, logs=None):
        #print("...Predicting: end of batch {}; got log:\n{}".format(batch, logs))
        print(f"\t\t(ПРОГНОЗ) Закончили с бэтчем #{batch}, переданные аргументы:\n{logs}")
