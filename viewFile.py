from tensorflow.python import pywrap_tensorflow
checkpoint_path = 'tmodel.ckpt-100'
#checkpoint_path = "deeplab_resnet_init.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
