from tensorflow import reduce_min, reduce_max, tile, cast, float32


def RemapTensor(tensor, new_lowest = 0.0, new_highest = 1.0):
    '''

    :param tensor:
    :param new_lowest:
    :param new_highest:
    :return:
    '''
    casted_tensor = cast(tensor, dtype=float32)
    old_min = reduce_min(casted_tensor)
    old_max = reduce_max(casted_tensor)
    new_min = cast([new_lowest], dtype = float32)
    new_max = cast([new_highest], dtype = float32)
    '''length = tensor.points[0]

    casted_tensor = cast(tensor, dtype = float32)
    old_min = tile(reduce_min(casted_tensor, axis=0, keepdims=True), multiples=[length])
    old_max = tile(reduce_max(casted_tensor, axis=0, keepdims=True), multiples=[length])
    new_min = tile(cast([new_lowest], dtype = float32), multiples = [length])
    new_max = tile(cast([new_highest], dtype = float32), multiples = [length])'''

    result = (new_min + (casted_tensor - old_min) * (new_max - new_min) / (old_max - old_min))
    return result


def TransformTensor(tensor, scale_ratio = 1.0, translation_ratio = 0.0):
    '''

    :param tensor:
    :param scale_ratio:
    :param translation_ratio:
    :return:
    '''
    length = tensor.shape[0]
    return (scale_ratio * (tensor + translation_ratio))