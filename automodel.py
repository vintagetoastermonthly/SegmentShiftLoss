from config import *
import numpy as np
#from data import *
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint #(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
from tensorflow.keras.callbacks import EarlyStopping #(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
from tensorflow.keras.callbacks import CSVLogger #(filename, separator=',', append=False)
from tensorflow.keras.callbacks import ReduceLROnPlateau #(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
from tensorflow.keras.optimizers import Adam
from keras_gradient_noise import add_gradient_noise
from PIL import Image,ImageFilter
from sklearn.utils import shuffle

NoisyAdam = add_gradient_noise(Adam)

def main():
	model=get_auto_model((height,width,1),auto_model_file)
	model.compile(
		loss='mse'
		,optimizer=NoisyAdam(learning_rate=0.001)
		,metrics=['mae']
	)
	print ( model.summary() )

	MySaver=ModelCheckpoint(auto_model_file, monitor='val_mae', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	MyCSV=CSVLogger('logs/auto.csv', separator=',', append=False)
	MyStopper=EarlyStopping(monitor='val_mae', min_delta=0, patience=75, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
	MyLR=ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=29, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	callbacks=[MySaver,MyCSV,MyStopper,MyLR]

	model.fit(
		getdata(train_label_dir+'/0/')
		,validation_data=getdata(valid_label_dir+'/0/')
		,steps_per_epoch=int(2048/batch_size)
		,validation_steps=int(2048/batch_size)
		,epochs=50
		,verbose=True
		,callbacks=callbacks
	)

	MySaver=ModelCheckpoint(auto_model_file, monitor='val_mae', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	MyCSV=CSVLogger('logs/auto.csv', separator=',', append=True)
	MyStopper=EarlyStopping(monitor='val_mae', min_delta=0, patience=75, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
	MyLR=ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=29, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	callbacks=[MySaver,MyCSV,MyStopper,MyLR]

	model.compile(
		loss='mae'
		,optimizer='adam'
		,metrics=['mae']
	)

	model.fit(
		getdata(train_label_dir+'/0/')
		,validation_data=getdata(valid_label_dir+'/0/')
		,steps_per_epoch=int(2048/batch_size)
		,validation_steps=int(2048/batch_size)
		,epochs=1000
		,verbose=True
		,callbacks=callbacks
	)

def blurim(im):
    s=int(np.random.random()*4+1)
    bim=np.array(im.filter(ImageFilter.GaussianBlur(s))).astype(np.float32)
    s=int(np.random.random()*120+30)
    bim=Image.fromarray(((bim>s)*255).astype(np.uint8))
    s=int(np.random.random()*10+1)
    bim=np.array(bim.filter(ImageFilter.GaussianBlur(s))).astype(np.float32)
    bim+=np.random.normal(0,np.max(im)/4,(height,width))
    bim[bim<0]=0
    bim[bim>255]=255
    return bim

def getdata(indir):
	filelist=os.listdir(indir)
	X=np.zeros((batch_size,height,width,o_channels),dtype=np.float32)
	y=np.zeros((batch_size,height,width,o_channels),dtype=np.float32)
	b=0
	while True:
		for f in shuffle(filelist):
			im=Image.open(indir+'/'+f).convert("L")
			bim=blurim(im)
			X[b,:,:,0]=bim/255.
			y[b,:,:,0]=np.array(im)/255.
			b+=1
			if b>=batch_size:
				yield X,y
				b=0

def get_auto_model(shape=(256,256,1),model_file=None):
	def down(x): 
		x=tf.keras.layers.Conv2D(filters=f,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(x)
		x=tf.keras.layers.BatchNormalization()(x)
		x=tf.keras.layers.SpatialDropout2D(0.5)(x)
		x=tf.keras.layers.Conv2D(filters=f,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(x)
		x=tf.keras.layers.BatchNormalization()(x)
		return x
	def up(x): 
		x=tf.keras.layers.Conv2DTranspose(filters=f,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(x)
		x=tf.keras.layers.BatchNormalization()(x)
		x=tf.keras.layers.SpatialDropout2D(0.5)(x)
		x=tf.keras.layers.Conv2D(filters=f,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(x)
		x=tf.keras.layers.BatchNormalization()(x)
		return x

	input_1 = tf.keras.layers.Input(shape=shape)
	f=shape[-1]*2
	if f < 32:
		f = 32
	fs=[]
	fs.append(f)
	x=down(input_1)
	while x.shape[1] > 32:
		#f=int(f*1.8+0.499)
		f*=2
		fs.append(f)
		x=down(x)
	f=fs.pop()
	while x.shape[1] < int(shape[0]/2):
		f=fs.pop()
		x=up(x)
	f=shape[-1]
	x=tf.keras.layers.Conv2DTranspose(filters=8,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(x)
	output_1 = tf.keras.layers.Conv2DTranspose(filters=f,kernel_size=(3,3),strides=(1,1),activation='sigmoid',padding='same')(x)

	model=tf.keras.models.Model(inputs=[input_1],outputs=[output_1])

	if model_file is not None and os.path.isfile(model_file):
		model.load_weights(model_file)

	return model

if __name__ == "__main__":
	"""
	gen=getdata(train_label_dir+'/0/')
	for X,y in gen:
		nim=np.zeros((256,512))
		nim[:,:256]=y[0,:,:,0]
		nim[:,256:]=X[0,:,:,0]
		Image.fromarray((nim*255).astype(np.uint8)).show()
		input("press enter to continue")
	"""
	main()
	print ( 'done.' )

