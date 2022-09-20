import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from config import *
from data import *
from model import *
from callbacks import *
from losses import *

threshold1=0.5
threshold2=0.3

border=int(height/4)

def main():
	print ( modelfile )
	model,age=getmodel(modelfile)
	automodel=tf.keras.models.load_model(auto_model_file)
	#model.load_weights('models/model.h5')

	print ( age, flush=True )

	for X,y in valid_generator:
		p=model.predict(X)
		a=automodel.predict(p)
		if modeltype=="DeepLab":
			X+=1
			X/=2
		nim=np.zeros((height,height*3,3),dtype=np.uint8)
		print ( X.shape, y.shape, np.min(X), np.mean(X), np.max(X), np.min(y), np.mean(y), np.max(y) )
		nim[:,:height,:]=X[0].reshape((height,height,3))*255
		nim[:,height:height*2,0]=(p[0,:,:,0].reshape((height,height)))*255
		nim[:,height:height*2,1]=nim[:,height:height*2,0]
		nim[:,height:height*2,2]=nim[:,height:height*2,0]

		nim[:,height*2:,0]=(a[0,:,:,0].reshape((height,height)))*255
		nim[:,height*2:,1]=nim[:,height*2:,0]
		nim[:,height*2:,2]=nim[:,height*2:,0]

		nim[border,border:height-border,:]=255-nim[border,border:height-border,:]
		nim[-border,border:height-border,:]=255-nim[-border,border:height-border,:]
		nim[border:-border,border,:]=255-nim[border:-border,border,:]
		nim[border:-border,height-border,:]=255-nim[border:-border,height-border,:]

		nim[border,height+border:height*2-border,:]=255-nim[border,height+border:height*2-border,:]
		nim[-border,height+border:height*2-border,:]=255-nim[-border,height+border:height*2-border,:]
		nim[border:-border,height+border,:]=255-nim[border:-border,height+border,:]
		nim[border:-border,height*2-border,:]=255-nim[border:-border,height*2-border,:]

		nim[border,height*2+border:-border,:]=127
		nim[-border,height*2+border:-border,:]=127
		nim[border:-border,height*2+border,:]=127
		nim[border:-border,-border,:]=127

		Image.fromarray(nim).show()
		input("press enter to continue") 
	

if __name__ == "__main__":
	main()
	print ( 'done.' )
