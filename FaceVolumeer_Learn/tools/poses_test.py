import tensorflow_graphics.geometry.transformation.euler
import transformations

from LoadTools import LoadDatasetGroundtruthData, LoadDatasetImageFilenames, LoadImagesByFilenames
import const
import numpy as np



def main():
    groundtruth_data = LoadDatasetGroundtruthData(const.DATASET_BASE_DIRECTORY, "AFLW2000", 0, 16)
    image_indices, image_projections, image_poses, image_shapes, \
    image_expressions, image_texture_params, image_lights = groundtruth_data
    image_filenames = LoadDatasetImageFilenames(const.DATASET_BASE_DIRECTORY, "AFLW2000", 0, 16)
    images = LoadImagesByFilenames(image_filenames)

    mean_proj = np.load(const.BASE_MODEL_DATA_DIRECTORY + "mean_projection.npy")
    std_proj = np.load(const.BASE_MODEL_DATA_DIRECTORY + "std_projection.npy")

    based_projections = (image_projections -  mean_proj) / std_proj

    projections = np.reshape(based_projections, (-1, 4, 2))
    m_row1 = projections[:, 0:3, 0] / np.linalg.norm(projections[:, 0:3, 0], axis=1, keepdims = True)  # tf.nn.l2_normalize(m[:, 0:3, 0], axis=1)
    m_row2 = projections[:, 0:3, 1] / np.linalg.norm(projections[:, 0:3, 1], axis=1, keepdims = True)  # tf.nn.l2_normalize(m[:, 0:3, 1], axis=1)
    m_row3 = np.cross(m_row1, m_row2)
    #m_row3 = np.pad(np.cross(m_row1, m_row2), [[0, 0], [0, 1]], mode='constant', constant_values=0)
    m_row3 = np.expand_dims(m_row3, axis=2)

    m = np.concatenate([projections[:, :3, :], m_row3], axis=2)

    angles = tensorflow_graphics.geometry.transformation.euler.from_rotation_matrix(m).numpy()


    print(angles)



if __name__=="__main__":
    main()