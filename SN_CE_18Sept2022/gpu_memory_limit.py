import tensorflow as tf
from config import *

#---------------------------------------------------------------------
# Memory Limit GPU
#---------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	try:
		tf.config.set_logical_device_configuration(
				gpus[0],
				[tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_MB)])
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)
#---------------------------------------------------------------------
