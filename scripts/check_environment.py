import platform
from datetime import datetime

import keras
import tensorflow as tf


print("VisionGuard | 20L-2189_A2_Ops")
print("Student: Yousaf Younas")
print("Checked:", datetime.now().strftime("%d-%m-%Y %I:%M %p"))
print("Operating system:", platform.platform())
print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible to TensorFlow:", len(gpus))

for gpu in gpus:
    print("Detected GPU:", gpu.name)