from config import *
from data import *
from model import *
from callbacks import *
from losses import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_gradient_noise import add_gradient_noise

NoisyAdam = add_gradient_noise(Adam)

def main():

	model,age=getmodel(modelfile)

	print ( age, flush=True )

	#adam=tf.keras.optimizers.Adam(learning_rate=0.001)
	adam=NoisyAdam(learning_rate=0.01)
	binIoU=tf.keras.metrics.BinaryIoU(target_class_ids=[1],threshold=0.5)

	border=int(height/4)
	IoU=CroppedBIoU(crop=((border,border),(border,border)))

	model.compile(
		optimizer=adam
		#,loss=focal_loss()
		,loss=tf.keras.losses.BinaryCrossentropy()
		#,loss=ce_jaccard_loss
		#,loss=entropy_weighted_bce
		#,loss=la_gauss_mse
		#,loss='mse'
		#,loss=segment_shift_bce
		#,loss=ce_dice_loss
		,metrics=['acc',IoU]
		#,metrics=['acc',ce_jaccard_loss2]
		,run_eagerly=True
	)

	print ( model.summary(), flush=True )

	if age != 'old':
		print ( 'starting initial training' )
	
		hist=model.fit(
			train_generator
			,steps_per_epoch=int(samples_per_epoch/batch_size)
			,validation_data=valid_generator
			,validation_steps=int(validation_samples_per_epoch/batch_size)
			,use_multiprocessing=False
			,workers=workers
			,callbacks=callbacks
			,epochs=initial_epochs
			,verbose=1
			,max_queue_size=128
		)
	
		print ( 'done initial training' )
	
		best_epoch=np.argmax(hist.history['val_acc'])
		val_acc=hist.history['val_acc'][best_epoch]
	
		print ( 'best epoch', best_epoch, 'best val_acc', val_acc )
	
	for layer in model.layers:
		layer.trainable=True

	adam=tf.keras.optimizers.Adam(learning_rate=0.00001)
	#adam=NoisyAdam(learning_rate=0.0001)

	model.compile(
		optimizer=adam
		#,loss=focal_loss()
		,loss=tf.keras.losses.BinaryCrossentropy()
		#,loss=ce_jaccard_loss
		#,loss=segment_shift_bce
		#,loss=jaccard_distance
		#,loss=la_gauss_mse
		#,loss=entropy_weighted_bce
		#,loss='mse'
		#,metrics=['acc']
		,metrics=['acc',IoU]
		#,metrics=['acc',binIoU]
		,run_eagerly=True
	)

	print ( model.summary(), flush=True )

	print ( 'starting initial training' )

	hist=model.fit(
		train_generator
		,steps_per_epoch=int(samples_per_epoch/batch_size)
		,validation_data=valid_generator
		,validation_steps=int(validation_samples_per_epoch/batch_size)
		,use_multiprocessing=False
		,workers=workers
		,callbacks=callbacks
		,epochs=epochs
		,verbose=1
		,max_queue_size=128
	)

	print ( 'done initial training' )

	best_epoch=np.argmax(hist.history['val_acc'])
	val_acc=hist.history['val_acc'][best_epoch]

	print ( 'best epoch', best_epoch, 'best val_acc', val_acc )

	print ( 'done.' )

if __name__ == "__main__":
	main()
