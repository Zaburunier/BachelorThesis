from tensorflow import function


@function
def ConvertToBase(value, mean_value, std_value):
    '''
    Вспомогательный метод преобразования абсолютной величины к базису распределения
    :param value: Значения
    :param mean_value: Среднее значение
    :param std_value: Стандартное отклонение
    :return: Базированные значения
    '''
    return (value - mean_value) / std_value


@function
def ConvertFromBase(value, mean_value, std_value):
    '''
    Вспомогательный метод преобразования величины, заданной в рамках распределения, к абсолютному значению
    :param value: Базированные
    :param mean_value: Среднее значение
    :param std_value: Стандартное отклонение
    :return: Абсолютные значения
    '''
    return mean_value + std_value * value