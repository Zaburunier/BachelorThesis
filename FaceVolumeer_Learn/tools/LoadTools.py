from os import listdir

from PIL import Image
from PIL.Image import BILINEAR
from numpy import fromfile, float32
from keras.preprocessing.image import img_to_array
#from keras.utils import img_to_array

from learning_const import DATASET_BASE_DIRECTORY, IMAGE_SIZE, TEXTURE_SIZE


def LoadDatasetImageFilenames(directory: str, dataset_name: str, offset, length):
    '''
    Загрузка списка файлов изображений для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список файлов
    '''
    filenames = LoadImageFilenamesFromDirectory(directory + "image/", dataset_name, offset = offset, length = length)

    print('Загружен список файлов с образцами датасета ' + dataset_name + ' в количестве ' + str(
        len(filenames)) + ' штук')
    return filenames


def LoadDatasetImages(directory: str, dataset_name: str):
    '''
    Загрузка изображений для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список текстур
    '''
    dataset_image_directory = directory + "image/" + dataset_name
    images = LoadImagesFromDirectory(dataset_image_directory)

    print('Загружены образцы датасета ' + dataset_name + ' в количестве ' + str(len(images)) + ' штук')
    return images


def LoadDatasetImageMasks(directory: str, dataset_name: str):
    '''
    Загрузка масок фотографий для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список масок текстур
    '''
    dataset_image_mask_directory = directory + "mask/image_mask/" + dataset_name
    image_masks = LoadImagesFromDirectory(dataset_image_mask_directory)

    print(
        'Загружены маски изображений для датасета ' + dataset_name + ' в количестве ' + str(len(image_masks)) + ' штук')
    return image_masks


def LoadDatasetTextureFilenames(directory: str, dataset_name: str, offset = 0, length = 0):
    '''
    Загрузка списка файлов текстур для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список текстур
    '''
    filenames = LoadImageFilenamesFromDirectory(directory + "texture/", dataset_name, offset = offset, length = length)

    print('Загружен список файлов с текстурами датасета ' + dataset_name + ' в количестве ' + str(
        len(filenames)) + ' штук')
    return filenames


def LoadDatasetTextures(directory: str, dataset_name: str):
    '''
    Загрузка текстур для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список текстур
    '''
    dataset_texture_directory = directory + "texture/" + dataset_name
    textures = LoadImagesFromDirectory(dataset_texture_directory)

    print('Загружены текстуры для датасета ' + dataset_name + ' в количестве ' + str(len(textures)) + ' штук')
    return textures


def LoadDatasetTextureMasks(directory: str, dataset_name: str):
    '''
    Загрузка масок текстур для датасета
    :param directory: Базовая папка данных
    :param dataset_name: Название датасета
    :return: Список масок текстур
    '''
    dataset_texture_mask_directory = directory + "mask/texture_mask/" + dataset_name
    texture_masks = LoadImagesFromDirectory(dataset_texture_mask_directory)

    print('Загружены маски текстур для датасета ' + dataset_name + ' в количестве ' + str(len(texture_masks)) + ' штук')
    return texture_masks


def LoadImageFilenamesFromDirectory(directory: str, dataset_name: str, include_dataset_directory=True,
                                    formats: [str] = None, offset = 0, length = 0) -> [str]:
    '''
    Вспомогательный метод для загрузки списка изображений в папке
    :param directory: Базовая папка с данными
    :param dataset_name: Название датасета
    :param include_dataset_directory: Включать ли в имя файла папку датасета
    :param formats: Допустимые форматы изображений (по умолчанию - png и jpg)
    :return: Список файлов с допустимыми форматами
    '''
    formats = ["jpg", "png"] if formats is None else formats

    image_filenames = listdir(directory + dataset_name)
    filenames = []

    for filename in image_filenames:
        if filename[filename.index(".") + 1:] in formats:
            filenames.append(dataset_name + "/" + filename)

    if (length == 0):
        length = len(filenames)

    return filenames[offset : offset + length]


def LoadImagesFromDirectory(directory: str) -> [str]:
    '''
    Вспомогательный метод для загрузки всех изображений в папке
    :param directory:
    :return:
    '''
    image_filenames = listdir(directory)
    images = []

    for filename in image_filenames:
        image = Image.open(directory + "/" + filename)
        image = image.resize(IMAGE_SIZE, resample=BILINEAR)
        images.append(image)

    return images


def LoadDatasetGroundtruthData(directory: str, dataset_name: str, offset = 0, length = 0):
    '''
    Загрузка вспомогательных данных для датасета с компьютера
    :param directory: Базовая папка с данными
    :param dataset_name: Название датасета
    :return: Подготовленные данные для каждого экземпляра: индексы (?), проекции, позы (?),
    формы, выражения лица (?), параметры текстур (?), освещение
    '''

    # В файлах структуры param.dat содержатся известные заранее данные для обучения
    fd = open(directory + 'filelist/' + dataset_name + '_param.dat')
    all_paras = fromfile(file=fd, dtype=float32)
    fd.close()

    # Данные в файлах организованы в виде последовательности цифр
    # Эти файлы были заготовлены заранее
    idDim = 1
    mDim = idDim + 8
    poseDim = mDim + 7
    shapeDim = poseDim + 199
    expDim = shapeDim + 29
    texDim = expDim + 40
    ilDim = texDim + 10
    # Преобразуем список чисел в таблицу, где каждая строка будет соответствовать параметрам
    # реконструкции, относящимся к конкретному экземпляру
    all_paras = all_paras.reshape((-1, ilDim)).astype(float32)

    if (length == 0):
        length = all_paras.shape[0] - offset

    image_indices = all_paras[offset: offset + length, 0:idDim]
    image_projections = all_paras[offset: offset + length, idDim:mDim]
    image_poses = all_paras[offset: offset + length, mDim:poseDim]
    image_shapes = all_paras[offset: offset + length, poseDim:shapeDim]
    image_expressions = all_paras[offset :offset + length, shapeDim:expDim]
    image_texture_params = all_paras[offset: offset + length, expDim:texDim]
    image_lights = all_paras[offset: offset + length, texDim:ilDim]

    print('Загружены лэйблы для датасета ' + dataset_name + ' в количестве ' + str(image_indices.shape[0]) + ' штук')

    return image_indices, image_projections, image_poses, image_shapes, \
           image_expressions, image_texture_params, image_lights


def LoadImagesByFilenames(filenames: [str]) -> [Image]:
    '''

    :param filenames:
    :return:
    '''
    images = []

    for filename in filenames:
        image = Image.open(DATASET_BASE_DIRECTORY + "image/" + filename)
        '''if (len(images) == 0):
            image.show()'''

        image = image.resize(IMAGE_SIZE, resample=BILINEAR)
        images.append(img_to_array(image))

    #print(f"По заданному списку файлов ({filenames}) загружено " + str(len(filenames)) + " изображений.")

    return images


def LoadTexturesByFilenames(filenames: [str]) -> [Image]:
    '''

    :param filenames:
    :return:
    '''
    textures = []

    for filename in filenames:
        image = Image.open(DATASET_BASE_DIRECTORY + "texture/" + filename)
        '''if (len(textures) == 0):
            image.show()'''
        #image = image.resize(TEXTURE_SIZE, resample=BILINEAR)
        textures.append(img_to_array(image))

    #print(f"По заданному списку файлов ({filenames}) загружено " + str(len(filenames)) + " текстур.")

    return textures


def LoadImageMasksByFilenames(filenames: [str]) -> [Image]:
    '''

    :param filenames:
    :return:
    '''
    images = []

    for filename in filenames:
        image = Image.open(DATASET_BASE_DIRECTORY + "mask/image_mask/" + filename)
        '''if (len(images) == 0):
            image.show()'''

        image = image.resize(IMAGE_SIZE, resample=BILINEAR)
        images.append(img_to_array(image))

    #("По заданному списку файлов загружено " + str(len(filenames)) + " масок изображений.")

    return images


def LoadTextureMasksByFilenames(filenames: [str]) -> [Image]:
    '''

    :param filenames:
    :return:
    '''
    textures = []

    for filename in filenames:
        image = Image.open(DATASET_BASE_DIRECTORY + "mask/texture_mask/" + filename)
        '''if (len(textures) == 0):
            image.show()'''

        #image = image.resize(TEXTURE_SIZE, resample=BILINEAR)
        textures.append(img_to_array(image))

    #print("По заданному списку файлов загружено " + str(len(filenames)) + " масок текстур.")

    return textures
