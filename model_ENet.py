"""
 * Author: David Knox
 * Created: 2020 Nov 14
 * Purpose: ENet based segmentation model
 * (c) Copyright by Lynker Analytics.
"""

from config import *
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4 as ENet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Conv2D,UpSampling2D,BatchNormalization as BN,SpatialDropout2D as Dropout,DepthwiseConv2D,Concatenate,Lambda,Reshape
from tensorflow.keras.utils import plot_model
from os.path import isfile
from losses import *

def getmodel(modelfile,shape=(512,512,3)):

	if isfile(modelfile):
		model,age=getmodel('empty_not_here',shape)
		#model.load_weights(modelfile,by_name=True, skip_mismatch=True)
		model.load_weights(modelfile)
		return model,'old'

	#-------------------------------------------------------------
	#  model component functions
	#-------------------------------------------------------------
	def compress(x,c):
	        return BN()(Conv2D(filters=c,kernel_size=(1,1),activation='relu',strides=(1,1),padding='same')(x))
	
	def up(x,c,activation='linear',B=True):
	        x=UpSampling2D(size=(2,2))(x)
	        x=Conv2D(filters=c,kernel_size=(3,3),activation=activation,strides=(1,1),padding='same')(x)
	        if B:
	                x=BN()(x)
	        return x
	
	def ires(x,c):
	        h=c*4
	        y=DepthwiseConv2D(h,kernel_size=(3,3),activation='relu',padding='same')(x)
	        y=BN()(y)
	        y=Conv2D(c,kernel_size=(1,1),activation='relu',padding='same')(y)
	        y=BN()(y)
	        return Add()([x,y])
	
	#-------------------------------------------------------------
	# Base ENet model use for encoder part of network
	#-------------------------------------------------------------
	basemodel=ENet(
		include_top=False
		,input_shape=shape
		,weights='imagenet'
	)
	
	#print ( basemodel.summary() )
	
	plot_model(basemodel,'enet.png',show_shapes=True)
	
	#-------------------------------------------------------------
	#  Build decoder part of network with skip connections
	#-------------------------------------------------------------
	h=16
	
	d16=basemodel.get_layer('top_activation').output
	d16=compress(d16,h)
	d16_32=up(d16,h)
	d16_64=up(d16_32,h)
	d16_128=up(d16_64,h)
	d16_256=up(d16_128,h)
	
	d32=basemodel.get_layer('block5a_expand_activation').output
	d32=compress(d32,h)
	d32=Concatenate()([d32,d16_32])
	d32=compress(d32,h)
	d32_64=up(d32,h)
	d32_128=up(d32_64,h)
	d32_256=up(d32_128,h)
	
	d64=basemodel.get_layer('block4a_expand_activation').output
	d64=compress(d64,h)
	d64=Concatenate()([d64,d32_64,d16_64])
	d64=compress(d64,h)
	d64_128=up(d64,h)
	d64_256=up(d64_128,h)
	
	d128=basemodel.get_layer('block3a_expand_activation').output
	d128=compress(d128,h)
	d128=Concatenate()([d128,d64_128,d32_128,d16_128])
	d128=compress(d128,h)
	d128_256=up(d128,h)
	
	d256=basemodel.get_layer('block2a_expand_activation').output
	d256=compress(d256,h)
	d256=Concatenate()([d256,d128_256,d64_256,d32_256,d16_256])
	d256=compress(d256,h)
	
	x=up(d256,h)
	
	outputs=Conv2D(filters=o_channels,kernel_size=(1,1),strides=(1,1),activation='sigmoid',padding='same')(x)

	model=Model(inputs=basemodel.inputs,outputs=outputs)
	
	for layer in basemodel.layers:
		layer.trainable=False
	
	#print ( model.summary() )
	plot_model(model,'LUenet.png',show_shapes=True)
	
	model.compile(
		optimizer='adam'
		,loss='mse'
	)

	return model, 'new'	

if __name__ == "__main__":
	model,age=getmodel(modelfile,shape=(512,512,3))
	print ( model.summary() )
