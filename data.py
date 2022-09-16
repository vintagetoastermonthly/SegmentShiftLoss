from config import *
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

seed=1

def pixel_rescale(im): return (im / 127.5) -1.
label_data_gen_parms = dict(
       	horizontal_flip=True
       	,vertical_flip=True
	,rescale=1./255.
)

if modeltype=='ENet':
	data_gen_parms = dict(
        	horizontal_flip=True
        	,vertical_flip=True
		,rescale=1./255.
	)
elif modeltype=='DeepLab':
	data_gen_parms = dict(
        	horizontal_flip=True
        	,vertical_flip=True
		,preprocessing_function=pixel_rescale
		#,rescale=1./255.
	)
else:
	print ( 'Unknown modeltype', modeltype )
	quit()

image_datagen = ImageDataGenerator(**data_gen_parms)
label_datagen = ImageDataGenerator(**label_data_gen_parms)

image_parms = dict(
        class_mode=None
        ,seed=seed
        ,shuffle=True
        ,batch_size=batch_size
        ,target_size=(height,width)
)
label_parms = dict(
        class_mode=None
        ,seed=seed
        ,shuffle=True
        ,batch_size=batch_size
        ,target_size=(height,width)
	,color_mode='grayscale'
)

train_image_generator = image_datagen.flow_from_directory(train_image_dir ,**image_parms)
train_label_generator = label_datagen.flow_from_directory(train_label_dir ,**label_parms)

valid_image_generator = image_datagen.flow_from_directory(valid_image_dir ,**image_parms)
valid_label_generator = label_datagen.flow_from_directory(valid_label_dir ,**label_parms)

fine_valid_image_generator = image_datagen.flow_from_directory(fine_valid_image_dir ,**image_parms)
fine_valid_label_generator = label_datagen.flow_from_directory(fine_valid_label_dir ,**label_parms)

train_generator = zip(train_image_generator,train_label_generator)
valid_generator = zip(valid_image_generator,valid_label_generator)

if __name__ == "__main__":
	samples=0
	for X,y in valid_generator:
		samples+=len(X)
		print ( samples, X.shape, y.shape, np.min(X), np.mean(X), np.max(X), np.min(y), np.mean(y), np.max(y) )
		""" 
		if modeltype=='DeepLab':
			X+=1
			X/=2
		im=(X[0].reshape((512,512,3))*255).astype(np.uint8)
		lbl=(y[0].reshape((512,512))*255).astype(np.uint8)
		Image.fromarray(im).show()
		Image.fromarray(lbl,"L").show()
		quit()
		""" 
		if samples > 5000: quit()
