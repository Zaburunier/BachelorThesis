IMAGE_SIZE = (224, 224)
TEXTURE_SIZE = (192, 224)
BATCH_SIZE = 8 # НЕ МЕНЯТЬ ИЗ-ЗА НАКОПЛЕНИЯ ГРАДИЕНТОВ В КЛАССЕ VOLUMEER2
GRADIENT_BATCH_SIZE = 16
LEARNING_RATE = 1e-03
RANDOM_SEED = 16

MESH_VERTICES = 53215  # Число вершин получаемой 3D-модели
MESH_TRIANGLES = 105840  # Число треугольников получаемой 3D-модели
ShapeParamDimensionSize = 160  # Число параметров кодового вектора формы
AlbedoParamDimensionSize = 160  # Число параметров кодового вектора альбедо
ProjectionParamDimensionSize = 8  # Число параметров кодового вектора ракурса
LightParamDimensionSize = 9  # Число параметров кодового вектора освещения

DATASET_BASE_DIRECTORY = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\data\\"
BASE_MODEL_DATA_DIRECTORY = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\3DMM_definition\\"
NETWORK_MODEL_SAVE_DIRECTORY = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\Trained model\\"
OPTIMIZER_SAVE_DIRECTORY = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\Trained optimizer\\"
EPOCH_COUNTER_SAVE_DIRECTORY = "D:\\Study\\Thesis\\FaceVolumeer_Learn\\Epoch count state\\"
NETWORK_MODEL_SAVE_NAME = "Volumeer (serialized)"

USING_LANDMARK_LOSS = True
