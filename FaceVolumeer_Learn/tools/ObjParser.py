import os
from numpy import ndarray
import logging
from pywavefront import Wavefront, configure_logging
from trimesh import load_mesh, Trimesh
from typing import Union

def ParseObjFile(path : str) -> Union[Trimesh, None]:
    '''
    Чтение файла в формате .obj
    :param path:
    :return:
    '''
    if (path.index('.') == -1 or path[path.index('.') + 1:] != "obj"):
        print("Некорретный формат, загрузка файла невозможна.")
        return None

    if (os.path.exists(path) == False):
        print("Некорретный путь, загрузка файла невозможна")
        return None

    configure_logging(
        logging.DEBUG,
        formatter=logging.Formatter('%(name)s-%(levelname)s: %(message)s')
    )

    #object = Wavefront(file_name = path, create_materials = True, collect_faces = True)
    object = load_mesh(path)
    return object
