import numpy as np

a = np.array([1,2,3])

print(a.dtype)
print(a.shape)

import tensorflow as tf; 

print(tf.config.list_physical_devices('GPU'))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


import os
from secret_keys import openai_api_key, azure_openai_api_key

print(openai_api_key)
print(azure_openai_api_key)