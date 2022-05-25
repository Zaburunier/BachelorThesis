import tensorflow as tf

def run():
    tf.config.threading.set_inter_op_parallelism_threads(16)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    physical_devices = tf.config.list_physical_devices('CPU')
    assert len(physical_devices) == 1, "No CPUs found"
    # Specify 2 virtual CPUs. Note currently memory limit is not supported.
    try:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration()])
        logical_devices = tf.config.list_logical_devices('CPU')
        assert len(logical_devices) == 2

        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration()])
    except:
        # Cannot modify logical devices once initialized.
        pass

    print(tf.config.list_logical_devices())