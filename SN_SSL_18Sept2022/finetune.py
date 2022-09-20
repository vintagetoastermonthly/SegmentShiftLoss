from config import *
from data import *
from model import *
from losses import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_gradient_noise import add_gradient_noise

NoisyAdam = add_gradient_noise(Adam)

monitor='val_loss'
MySaver=tf.keras.callbacks.ModelCheckpoint(modelfile, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
MyCSV=tf.keras.callbacks.CSVLogger(csvfile, separator=',', append=False)
MyStopperFinetune=tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=earlystop_patience_finetune, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
finetune_callbacks=[MySaver,MyCSV,MyStopperFinetune]

def main():

	model,age=getmodel(modelfile)

	if age != 'old':
		print ( 'No old model found', modelfile )
		quit()

	adam=NoisyAdam(learning_rate=0.001)
	binIoU=tf.keras.metrics.BinaryIoU(target_class_ids=[1],threshold=0.5)

	for layer in model.layers:
		layer.trainable=True

	#adam=tf.keras.optimizers.Adam(learning_rate=0.00001)
	learning_rate=0.0001
	while learning_rate > 1e-7:

		print ( 'learning rate', learning_rate )

		adam=NoisyAdam(learning_rate=learning_rate)

		#We want to not reset the saver with each loop but we do want to reset the stopper - otherwise it might not stop if there is no improvement in a loop.
		MyStopperFinetune=tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=earlystop_patience_finetune, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
		finetune_callbacks=[MySaver,MyCSV,MyStopperFinetune]

		model.compile(
			optimizer=adam
			#,loss=focal_loss()
			,loss=tf.keras.losses.BinaryCrossentropy()
			#,loss=la_gauss_mse
			#,metrics=['acc']
			,metrics=['acc',binIoU]
		)
	
		hist=model.fit(
			train_generator
			,steps_per_epoch=int(samples_per_epoch/batch_size)
			,validation_data=valid_generator
			,validation_steps=int(validation_samples_per_epoch/batch_size)
			,use_multiprocessing=False
			,workers=workers
			,callbacks=finetune_callbacks
			,epochs=epochs
			,verbose=1
			,max_queue_size=128
		)

		print ( 'done training at learning_rate:', learning_rate )

		best_epoch=np.argmax(hist.history['val_acc'])
		val_acc=hist.history['val_acc'][best_epoch]

		print ( 'best epoch', best_epoch, 'best val_acc', val_acc )

		learning_rate/=10.
		model,age=getmodel(modelfile)

	print ( 'done.' )

if __name__ == "__main__":
	main()
