def PrepareInputData(self, data: dict):
    '''
    Подготовка входных данных (если они организованы в виде словаря с доп. информацией)
    :param data: Словарь входных данных
    (допустимые ключи: "image", "points", "albedo", "projection", "light", "unwarped_textures")
    :return: Словарь подготовленных входных данных
    '''
    if (isinstance(data, dict) == False):
        print("Для подготовки входных данных требуется передача словаря."
              "\nПодготовка невозможна, действие проигнорировано")
        return data

    if (len(data.keys()) == 1 and data.get("image") != None):
        print("Для подготовки входных данных передан словарь без дополнительных данных."
              "\nПодготовка бессмысленна, действие проигнорировано")
        return data

    return data
