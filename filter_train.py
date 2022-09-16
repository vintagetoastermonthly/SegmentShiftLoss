import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from config import *
from data import *
from model import *
from callbacks import *
from losses import *
from os import listdir
from os.path import isfile
from PIL import Image


threshold=0.5
iou_threshold=0.9

def main():
	print ( modelfile )
	model,age=getmodel(modelfile)
	#model.load_weights('models/model.h5')

	print ( age, flush=True )

	for f in listdir(train_image_dir+'/0/'):
		imfile=train_image_dir+'/0/'+f
		lbfile=train_label_dir+'/0/'+f
		if isfile(imfile) and isfile(lbfile):
			#print ( imfile )
			X=(np.array(Image.open(imfile))/127.5 -1.).reshape((1,height,height,3))
			y=((np.array(Image.open(lbfile))>0).astype(np.float32)).reshape((1,height,height,3))[:,:,:,0]

			p=model.predict(X)
	
			yim=y[0].reshape((height,height))>0
			pim=p[0,:,:,0].reshape((height,height))>threshold

			#iou = true_positives / (true_positives + false_positives + false_negatives)
			tp = np.sum(( pim == 1 ) & (yim == 1 ))
			fp = np.sum(( pim == 1 ) & (yim == 0 ))
			fn = np.sum(( pim == 0 ) & (yim == 1 ))
			#print ( 'tp', tp, 'fp', fp, 'fn', fn )
			#tn = ( pim < 0 ) == (yim < 0 )
	
			iou=tp / (tp + fp + fn )
			print ( 'filenames:', imfile, ':', lbfile, 'IoU:', iou, flush=True )
			if iou < iou_threshold:
				print ( 'del ',imfile.replace('//','/').replace('/','\\') )
				print ( 'del ',lbfile.replace('//','/').replace('/','\\') )

			""" 
			Image.open(imfile).show()
			Image.fromarray(pim*255).show()
			input("press enter to continue") 
			""" 
	

if __name__ == "__main__":
	main()
	print ( 'done.' )
