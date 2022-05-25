from tensorflow import convert_to_tensor, config, load_op_library, load_library, reshape
from numpy import ndarray, float32
import ProjectionTools
import math
from transformations import compose_matrix


def main():
    data1 = [[0.3, 10.2, -369.32456], [math.radians(15.23743), math.radians(90.0), math.radians(-10.3402)], [1.0, 1.0, 1.0]]
    print_matrices(data1)
    print_matrices([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def print_matrices(data):
    compose_matrix_value = compose_matrix(translate = data[0], angles = data[1], scale = data[2])
    view_matrix_value = ProjectionTools.view_matrix_mapped(convert_to_tensor(data))
    normalizing_ratio = ProjectionTools.CalculateTfgViewMatrices(reshape(convert_to_tensor(data[1] + [0.0] + data[0] + [0.0]), [1, 8]))
    print(f"Got data:\n{data}\nvalues through compose_matrix:\n{compose_matrix_value}"
          f"\nvalue through view_matrix_mapped:\n{view_matrix_value}"
          f"\nvalue throught tfg extension:\n{normalizing_ratio}")


if __name__=="__main__":
    config.run_functions_eagerly(True)
    main()
    #load_library("C:/Users/lenya/Downloads/rasterizer_op.so")#("D:\\Anaconda\\envs\\AI_3D\\Lib\\site-packages\\tensorflow_graphics\\rendering\\opengl\\rasterizer_op.so")