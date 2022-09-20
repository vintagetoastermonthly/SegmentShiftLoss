#retrieved from: https://github.com/keras-team/keras-io/blob/master/examples/vision/deeplabv3_plus.py
#on 27 Sept 2021
#modified dknox  27 Sept 2021
"""
ResNet50					ENet B4
conv4_block6_2_relu	(None, 32, 32, 256)	block6a_expand_activation	(None, 32, 32, 960)
conv2_block3_2_relu	(None, 128, 128, 64)	block3a_expand_activation	(None, 128, 128, 192)
"""
"""
Title: Multiclass semantic segmentation using DeepLabV3+
Author: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/08/31
Last modified: 2021/09/1
Description: Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation.
"""
"""
## Introduction

Semantic segmentation, with the goal to assign semantic labels to every pixel in an image,
is an essential computer vision task. In this example, we implement
the **DeepLabV3+** model for multi-class semantic segmentation, a fully-convolutional
architecture that performs well on semantic segmentation benchmarks.

### References:

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
"""

import numpy as np
from os.path import isfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import *


def getmodel(modelfile,shape=(512,512,3),o_channels=1):
	
	"""
	## Building the DeepLabV3+ model
	
	DeepLabv3+ extends DeepLabv3 by adding an encoder-decoder structure. The encoder module
	processes multiscale contextual information by applying dilated convolution at multiple
	scales, while the decoder module refines the segmentation results along object boundaries.
	
	![](https://github.com/lattice-ai/DeepLabV3-Plus/raw/master/assets/deeplabv3_plus_diagram.png)
	
	**Dilated convolution:** With dilated convolution, as we go deeper in the network, we can keep the
	stride constant but with larger field-of-view without increasing the number of parameters
	or the amount of computation. Besides, it enables larger output feature maps, which is
	useful for semantic segmentation.
	
	The reason for using **Dilated Spatial Pyramid Pooling** is that it was shown that as the
	sampling rate becomes larger, the number of valid filter weights (i.e., weights that
	are applied to the valid feature region, instead of padded zeros) becomes smaller.
	"""
	
	
	def convolution_block(
		block_input,
		num_filters=256,
		kernel_size=3,
		dilation_rate=1,
		padding="same",
		use_bias=False,
	):
		x = layers.Conv2D(
			num_filters,
			kernel_size=kernel_size,
			dilation_rate=dilation_rate,
			padding="same",
			use_bias=use_bias,
			kernel_initializer=keras.initializers.HeNormal(),
		)(block_input)
		x = layers.BatchNormalization()(x)
		return tf.nn.relu(x)
	
	
	def DilatedSpatialPyramidPooling(dspp_input):
		dims = dspp_input.shape
		x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
		x = convolution_block(x, kernel_size=1, use_bias=True)
		out_pool = layers.UpSampling2D(
			size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
		)(x)
	
		out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
		out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
		out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
		out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
	
		x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
		output = convolution_block(x, kernel_size=1)
		return output
	
	
	"""
	The encoder features are first bilinearly upsampled by a factor 4, and then
	concatenated with the corresponding low-level features from the network backbone that
	have the same spatial resolution. For this example, we
	use a ResNet50 pretrained on ImageNet as the backbone model, and we use
	the low-level features from the `conv4_block6_2_relu` block of the backbone.
	"""
	
	
	def DeeplabV3Plus(image_size, num_classes):
		model_input = keras.Input(shape=(image_size, image_size, 3))
		basemodel = keras.applications.ResNet50(
			weights="imagenet", include_top=False, input_tensor=model_input
		)
		for layer in basemodel.layers:
			layer.trainable=False
			if layer.name=='conv1_bn':
				layer.trainable=True

		x = basemodel.get_layer("conv4_block6_2_relu").output
		x = DilatedSpatialPyramidPooling(x)
	
		input_a = layers.UpSampling2D(
			size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
			interpolation="bilinear",
		)(x)
		input_b = basemodel.get_layer("conv2_block3_2_relu").output
		input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
	
		x = layers.Concatenate(axis=-1)([input_a, input_b])
		x = convolution_block(x)
		x = convolution_block(x)
		x = layers.UpSampling2D(
			size=(image_size // x.shape[1], image_size // x.shape[2]),
			interpolation="bilinear",
		)(x)
		model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid', padding="same")(x)
		return keras.Model(inputs=model_input, outputs=model_output)
	
	
	model = DeeplabV3Plus(image_size=width, num_classes=o_channels)
	age='new'
	if isfile(modelfile):
		model.load_weights(modelfile)
		age='old'
	return model, age
	
