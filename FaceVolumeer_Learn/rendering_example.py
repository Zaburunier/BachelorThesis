import numpy as np
import tensorflow as tf

from network.VolumeerNetwork import *
from rendering.RenderingTools import ComputeNormals, save_images
from rendering.Warping import *

VERTEX_NUM = 53215


def main():
    batch_size = 16
    output_size = 224
    texture_size = [192, 224]
    mDim = 8
    vertexNum = VERTEX_NUM
    channel_num = 3


    data = np.load('D:/Study/Thesis/FaceVolumeer_Learn/TF_newop/sample_data.npz')
    texture_ph = data['sample_texture'][:4, :, :, :]
    shape_ph = tf.reshape(data['sample_shape'], shape = [batch_size, -1, 3])[:4, :, :]
    m_ph = tf.cast(data['sample_m'], dtype = tf.float32)[:4, :]

    volumeer = VolumeerNetwork()
    #gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0", allow_growth=True)

    """ Graph """

    projection_matrices = rendering.ProjectionTools.CalculateProjectionMatrices(m_ph)
    vertex_normals, triangle_normals = ComputeNormals(vertices=shape_ph,
                                                      triangles=volumeer.base_model_triangle_data.data,
                                                      vertex_triangle_map=volumeer.base_model_triangle_vertex_map_data.data)
    vertex_normals = ProjectionTools.RotateVectors(vectors=vertex_normals,
                                                   projection_matrices=projection_matrices)
    triangle_normals = ProjectionTools.RotateVectors(vectors=triangle_normals,
                                                     projection_matrices=projection_matrices)

    images, foreground_mask = Warping.Warp(volumeer.base_model_triangle_data.data,
                                           volumeer.base_model_vertex_uv_map_data,
                                           texture_ph, shape_ph, triangle_normals, output_size=output_size)

    save_images(images, [4, -1], './rendered_img.png')
    save_images(texture_ph, [4, -1], './unwarped_textures.png')

    return

    '''with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                                              gpu_options=gpu_options)) as sess:
        """ Graph """
        m_ph = tf.compat.v1.placeholder(tf.float32, [batch_size, mDim])
        shape_ph = tf.compat.v1.placeholder(tf.float32, [batch_size, vertexNum * 3])
        texture_ph = tf.compat.v1.placeholder(tf.float32, [batch_size] + texture_size + [channel_num])

        vertices = reshape(shape_ph, [BATCH_SIZE, -1, 3])
        projection_matrices = rendering.ProjectionTools.CalculateProjectionMatrices(m_ph)
        vertex_normals, triangle_normals = ComputeNormals(vertices=shape_ph,
                                                          triangles=volumeer.base_model_triangle_data.data,
                                                          vertex_triangle_map=volumeer.base_model_triangle_vertex_map_data.data)
        vertex_normals = ProjectionTools.RotateVectors(vectors=vertex_normals,
                                                       projection_matrices=projection_matrices)
        triangle_normals = ProjectionTools.RotateVectors(vectors=triangle_normals,
                                                         projection_matrices=projection_matrices)

        images, foreground_mask = Warping.Warp(volumeer.base_model_triangle_data.data, volumeer.base_model_vertex_uv_map_data,
                                               texture_ph, vertices, texture_ph, output_size=output_size)

        s_img = sess.run(images, feed_dict={texture_ph: data['sample_texture'], shape_ph: data['sample_shape'],
                                            m_ph: data['sample_m']})

        save_images(s_img, [4, -1], './rendered_img.png')
        save_images(data['sample_texture'], [4, -1], './unwarped_textures.png')'''


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    main()
