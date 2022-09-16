eta=1e-9

#focal loss from here: https://github.com/mkocabas/focal-loss-keras
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

def focal_loss(gamma=2., alpha=.25):
		def focal_loss_fixed(y_true, y_pred):
				eps=1e-7
				y_pred = K.clip(y_pred, eps, 1-eps)
				pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
				pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
				return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
		return focal_loss_fixed

#this is a workaround for loading existing models
def focal_loss_fixed(y_true, y_pred):
		gamma=2.
		alpha=.25
		eps=1e-7
		y_pred = K.clip(y_pred, eps, 1-eps)
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def la_gauss_mse(y_true, y_pred, filter_shape=15 ,sigma=3):
	y_true=tfa.image.gaussian_filter2d(y_true ,filter_shape=filter_shape ,sigma=sigma)
	y_pred=tfa.image.gaussian_filter2d(y_pred ,filter_shape=filter_shape ,sigma=sigma)
	squared_difference = tf.square(y_true - y_pred)
	return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1

def segment_shift_mse(y_true, y_pred, shift=10, stride=5, blur=True, filter_shape=7, sigma=1):  #shift and stride in pixels
	shift+=1
	#expects y_true and y_pred to have shape:  (batch_size,height,width,channels)
	if blur:
		y_true=tfa.image.gaussian_filter2d(y_true ,filter_shape=filter_shape ,sigma=sigma)
		y_pred=tfa.image.gaussian_filter2d(y_pred ,filter_shape=filter_shape ,sigma=sigma)
	
	#exclude border of width shift from images to consider only overlapping regions with shift tolerance
	y_true=y_true[:,shift:-shift,shift:-shift]
	y_pred=y_pred[:,shift:-shift,shift:-shift]
	
	(batch_size,h,w,c)=y_true.numpy().shape

	return_mse=[]
 
	for b in range(batch_size):
		lowest_pmse=9e6
		return_mse.append([])
		for i in range(-shift+1,+shift,stride):
			for j in range(-shift+1,+shift,stride):
				p=y_pred[b,shift:-shift,shift:-shift]
				t=y_true[b,(shift+i):-(shift-i),(shift+j):-(shift-j)]
				mse=tf.math.reduce_mean(tf.square(t - p), axis=-1)  # Note the axis=-1
				#print ( t.shape, p.shape, i,j, np.array(tf.math.reduce_mean(mse)) )
				patch_mse=tf.math.reduce_mean(mse)
				if patch_mse < lowest_pmse:
					lowest_pmse=patch_mse
					return_mse[-1]=mse
	return_mse_t=tf.stack(return_mse)
	paddings = tf.constant([[0, 0,], [2*shift, 2*shift], [2*shift, 2*shift]]) 
	
	return tf.pad(return_mse_t, paddings, "CONSTANT", constant_values=0)

def segment_shift_bce_OLD(y_true, y_pred, shift=10, stride=5):  #shift and stride in pixels
	shift+=1
	#expects y_true and y_pred to have shape:  (batch_size,height,width,channels)
	#exclude border of width shift from images to consider only overlapping regions with shift tolerance
	#y_true=y_true[:,shift:-shift,shift:-shift]
	#y_pred=y_pred[:,shift:-shift,shift:-shift]

	(batch_size,h,w,c)=y_true.numpy().shape

	return_bce=[]
	
	bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

	for b in range(batch_size):
		lowest_loss=9e6
		return_bce.append([])
		for i in range(-shift+1,+shift,stride):
			for j in range(-shift+1,+shift,stride):
		#Should we be shifting the truth or the prediction?
		#discuss!
		#At first I moved the truth, thinking about pixel loss position.
		#then, I realised that the loss is applied to all pixels not per pixel
		#And, the truth should be the truth. It's the prediction that needs to be found that matches the truth
		#So, shift the prediction.
				t=y_true[b,shift:-shift,shift:-shift]
				p=y_pred[b,(shift+i):-(shift-i),(shift+j):-(shift-j)]
				loss=bce(t,p)
				#print ( t.shape, p.shape, i,j, loss.numpy() ) #, np.array(tf.math.reduce_mean(mse)) )
				if loss < lowest_loss:
					lowest_loss=loss
					return_bce[-1]=loss
	return_bce_t=tf.stack(return_bce)
	#paddings = tf.constant([[0, 0,], [2*shift, 2*shift], [2*shift, 2*shift]])

	return return_bce_t #tf.pad(return_bce_t, paddings, "CONSTANT", constant_values=0)

def segment_shift_bce(y_true, y_pred, shift=10, stride=5):  #shift and stride in pixels
	shift+=1

	(batch_size,h,w,c)=y_true.numpy().shape

	bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
	
	Ts=[]
	Ps=[]

	for b in range(batch_size):
		t=y_true[b,shift:-shift,shift:-shift]
		for i in range(-shift+1,+shift,stride):
			for j in range(-shift+1,+shift,stride):
				p=y_pred[b,(shift+i):-(shift-i),(shift+j):-(shift-j)]
				Ts.append(t)
				Ps.append(p)
	losses=tf.reduce_mean(bce(tf.stack(Ts),tf.stack(Ps)),axis=[1,2])
	losses=tf.reshape(losses,(batch_size,losses.shape[0]//batch_size))
	losses=tf.reduce_min(losses,axis=1)

	return tf.reduce_mean(losses)

def segment_shift_focal(y_true, y_pred, shift=10, stride=5):  #shift and stride in pixels
	shift+=1

	(batch_size,h,w,c)=y_true.numpy().shape

	bce = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, reduction="none")
	
	Ts=[]
	Ps=[]

	for b in range(batch_size):
		t=y_true[b,shift:-shift,shift:-shift]
		for i in range(-shift+1,+shift,stride):
			for j in range(-shift+1,+shift,stride):
				p=y_pred[b,(shift+i):-(shift-i),(shift+j):-(shift-j)]
				Ts.append(t)
				Ps.append(p)
	losses=tf.reduce_mean(bce(tf.stack(Ts),tf.stack(Ps)),axis=[1,2])
	losses=tf.reshape(losses,(batch_size,losses.shape[0]//batch_size))
	losses=tf.reduce_min(losses,axis=1)

	return tf.reduce_mean(losses)

def segment_shift_hybrid(y_true, y_pred, shift=10, stride=5):  #shift and stride in pixels
	shift+=1

	(batch_size,h,w,c)=y_true.numpy().shape

	Ts=[]
	Ps=[]

	for b in range(batch_size):
		t=y_true[b,shift:-shift,shift:-shift]
		for i in range(-shift+1,+shift,stride):
			for j in range(-shift+1,+shift,stride):
				p=y_pred[b,(shift+i):-(shift-i),(shift+j):-(shift-j)]
				Ts.append(t)
				Ps.append(p)
	#losses=tf.reduce_mean(ce_jaccard_loss2(tf.stack(Ts),tf.stack(Ps)),axis=[1,2])
	losses=ce_jaccard_loss2(tf.stack(Ts),tf.stack(Ps))
	losses=tf.reshape(losses,(batch_size,losses.shape[0]//batch_size))
	losses=tf.reduce_min(losses,axis=1)

	return tf.reduce_mean(losses)

def loss_with_nulls(lossfn,y,p,**kwargs):
        l=lossfn(y,p,**kwargs)
        nans=tf.reduce_sum(tf.cast(tf.math.is_nan(l),l.dtype))
        tots=tf.cast(tf.reduce_prod(l.shape),l.dtype)
        s=(tots)/(tots-nans+0.0001)
        n=tf.where(tf.math.is_nan(l), tf.zeros_like(l), l)
        return tf.reduce_mean(n * s)
"""

def entropy_weighted_bce(y_true, y_pred):
	bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")

	loss=bce(y_true,y_pred)
	weight=bce(y_pred,y_pred)

	wl=tf.math.sqrt(loss*weight)

	return 0.9*wl + 0.1*loss
"""

def entropy_weighted_bce(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")

        loss=bce(y_true,y_pred)
        weight=bce(y_pred,y_pred)

        l=tf.math.sqrt(loss*weight)
        
        l/=tf.reduce_mean(l)
        l*=tf.reduce_mean(loss)
        
        nans=tf.reduce_sum(tf.cast(tf.math.is_nan(l),l.dtype))
        n=tf.where(tf.math.is_nan(l), tf.zeros_like(l), l)

        return 0.9*n + 0.1*loss

def CroppedBIoU(crop=((128,128),(128,128))):
	#binIoU=tf.keras.metrics.BinaryIoU(target_class_ids=[1],threshold=0.5)
	#crop = ((top_crop, bottom_crop), (left_crop, right_crop))
	def BIoU(y,p):
		y_true=tf.keras.layers.Cropping2D(cropping=crop)(y)
		y_pred=tf.keras.layers.Cropping2D(cropping=crop)(p)

		intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
		union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
		IoU = intersection / (union + K.epsilon())

		#return binIoU(y_true,y_pred)
		return IoU
	return BIoU

def ce_jaccard_loss(y_true, y_pred):
	ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

	intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
	union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
	jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
	loss = ce_loss + jaccard_loss
	return loss

from keras import backend as K


def jaccard_distance(y_true, y_pred, smooth=100):
	"""Jaccard distance for semantic segmentation.

	Also known as the intersection-over-union loss.

	This loss is useful when you have unbalanced numbers of pixels within an image
	because it gives all classes equal weight. However, it is not the defacto
	standard for image segmentation.

	For example, assume you are trying to predict if
	each pixel is cat, dog, or background.
	You have 80% background pixels, 10% dog, and 10% cat.
	If the model predicts 100% background
	should it be be 80% right (as with categorical cross entropy)
	or 30% (with this loss)?

	The loss has been modified to have a smooth gradient as it converges on zero.
	This has been shifted so it converges on 0 and is smoothed to avoid exploding
	or disappearing gradient.

	Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
			= sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

	# Arguments
		y_true: The ground truth tensor.
		y_pred: The predicted tensor
		smooth: Smoothing factor. Default is 100.

	# Returns
		The Jaccard distance between the two tensors.

	# References
		- [What is a good evaluation measure for semantic segmentation?](
		   http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)

	"""
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
	jac = (intersection + smooth) / (sum_ - intersection + smooth)
	return (1 - jac) * smooth

def ce_jaccard_loss2(y_true, y_pred):
	ce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred),axis=(1,2))

	intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1,2,3))
	union = tf.reduce_sum(y_true,axis=(1,2,3)) + tf.reduce_sum(y_pred,axis=(1,2,3)) - intersection
	jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
	loss = ce_loss + jaccard_loss
	return loss

def ce_dice_loss(y_true, y_pred):
	ce_loss = binary_crossentropy(y_true, y_pred)

	intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
	union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
	dice_loss = - tf.log((intersection + K.epsilon()) / (union + K.epsilon()))
	loss = ce_loss + dice_loss
	return loss

