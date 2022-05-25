from tensorflow import  gather, reshape, gather_nd, concat, split, float32, convert_to_tensor, tile, function

import learning_const
from network.loss.L2Loss import L2Loss
from keras.losses import Loss
from data.base_model.LandmarkData import LandmarkData


class LandmarkLoss(L2Loss):
    '''
    Функция потерь по расположению ключевых точек
    Эта метрика используется на этапе обучения, где ключевые точки исходных моделей известны,
    а ключевые точки создаваемых изображений можно, зная деформацию исходника, подсчитать самостоятельно
    '''

    def __init__(self, weight = 1.0):
        super(LandmarkLoss, self).__init__(weight)
        self.landmarks_indices = reshape(convert_to_tensor(LandmarkData(learning_const.BASE_MODEL_DATA_DIRECTORY + "3DMM_keypoints.dat").data), (1, 68))
        #self.landmarks_indices = convert_to_tensor(LandmarkData(const.BASE_MODEL_DATA_DIRECTORY + "3DMM_keypoints.dat").data)


    @function
    def GatherLandmarks(self, vertices):
        #return gather_nd(vertices, self.landmarks_indices, batch_dims = 1)
        return gather(vertices, tile(self.landmarks_indices, (vertices.shape[0], 1)), batch_dims = 1)