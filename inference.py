from config import *
#indir='C:/Users/support/Desktop/LINZ_Buildings/test/'
#indir='C:/Users/support/Desktop/LINZ_Buildings/buffered_area3_canterbury_0_3m_2019/'
#indir='D:/DataArchive/enhanced_area3_canterbury_0_3m_2019/'
#indir='D:/DataArchive/redo_area2/'
#indir='D:/DataArchive/buffered_area5_canterbury_waitaki_0_3m_2021/'
#indir='C:/Users/support/Desktop/FAST_DATA/AREA3_Subset_enhanced/'
#indir='C:/Users/support/Desktop/FAST_DATA/buffered_area5_canterbury_waitaki_0_3m_2021/'
#indir='D:/DataArchive/waitaki_urban_2020-2021_0.075m_RGB/'
#indir='C:/Users/support/Desktop/FAST_DATA/enhanced_SUB_buffered_raw_imagery_area3_canterbury_0_3m_2019/'
#indir='D:/DataArchive/buffered_area4_canterbury_0_2m_2020_2021/'
#indir='C:/Users/support/Desktop/FAST_DATA/Area4_Sub/'
#indir='D:/DataArchive/buffered_area7_auckland_0_075m_2020/'
#indir='C:/Users/support/Desktop/FAST_DATA/buffered_area7_auckland_0_075m_2020/'
#indir='D:/DataArchive/Area9/buffered_area9/'
indir='indata/'

import numpy as np
import rasterio
from rasterio.windows import Window	#Window(col_off, row_off, width, height)
from model import *
from os import listdir
from os.path import isfile
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.exposure import adjust_gamma

from gpu_memory_limit import *

infiles=[]
for f in listdir(indir):
	exts=f.split('.')[-1]
	if 'tif' in exts or 'jp2' in exts:
		infiles.append(indir+f)

#stride=256
stride=int(height/2)
#stride=192
border=int((height-stride)/2)

#model3,age=getmodel(publishfile)
#model,age=getmodel('models/Impervious_10cm_rgb_512_V2.h5')
model,age=getmodel(modelfile,shape=(height,width,3))
#automodel=tf.keras.models.load_model(auto_model_file)
for layer in model.layers:
	layer.trainable=False
model.compile(optimizer='sgd' ,loss='mse')

print ( model.summary() )

def main():

	for inraster in infiles:
		outraster=inraster.replace(indir,'output/').replace('.jp2','.tif')
		#autooutraster=inraster.replace(indir,'output/auto/').replace('.jp2','.tif')
		#if 'BY21_10000_0202' not in inraster: continue
		if isfile(outraster):
			print ( outraster, 'already exists' )
			continue
		print ( 'Processing', outraster )

		chips=[]
		coords=[]
	
		def infer(dst):
			if modeltype=='DeepLab':
				print ( 'is DEEPLAB' )
				X=np.array(chips)/127.5 -1
			else:
				print ( 'not DEEPLAB' )
				X=np.array(chips)/255.
			print ( X.shape )
			p1=model.predict(X) #*255
			p1*=255
			#p1=np.argmax(p1,axis=-1).astype(np.uint8)
			
			for i in range(len(coords)):
				(c,r)=coords[i]
				try:
					dst.write(p1[i].reshape(height,width)[border:-border,border:-border],window=Window(c+border,r+border,stride,stride),indexes=1)
				except Exception as e:
					print ( 'Exception', e )
	
		with rasterio.open(inraster,'r') as src:
			profile=src.profile
			meta=src.meta
			profile.update({
				'count':1,
				'bigtiff':True,
				'driver': 'GTiff',
				'nodata': 255,
				'compress': 'LZW',
				'dtype': 'uint8'
			})
			del profile['nodata']
			print ( profile )
			with rasterio.open(outraster,'w',**profile) as dst:
				#with rasterio.open(autooutraster,'w',**profile) as autodst:
				if True:
					print ( 'hello world' )
					b=0
					for col_offset in range(0,meta['width'],stride):
						print ( '', flush=True )
						for row_offset in range(0,meta['height'],stride):
							print ( col_offset, '/', meta['width'], row_offset, '/', meta['height'], flush=True )
							im=np.zeros((height,width,3),dtype=np.float32)
							sizeOK=True
							for c in range(3):
								patch=src.read(c+1,window=Window(col_offset,row_offset,width,height))
								if patch.shape[:2] == (height,width):
									im[:,:,c]=patch
									#print ( 'happy shape', patch.shape )
								else:
									sizeOK=False
									#print ( 'irregular shape', patch.shape )
							if sizeOK:
								if np.max(im) > 0:
									coords.append((col_offset,row_offset))
									#chips.append(adjustim(im))
									chips.append(im)
									b+=1
									if b >= batch_size:
										infer(dst)
										b=0
										chips=[]
										coords=[]
								else:
									print ( 'blank image', np.min(im), np.max(im) )
		
					if len(chips) > 0:
						infer(dst)

def adjustim(rgb):
	#---------------------------------------------------------------
	# convert to LAB colour space
	# and apply gamma correction to luminance channel
	#---------------------------------------------------------------
	lab=np.array(rgb2lab(rgb.astype(np.uint8)))
	l=lab[:,:,0]
	l=adjust_gamma(l,0.5,1)

	#---------------------------------------------------------------
	# Apply a min-max stretch on luminance channel
	#---------------------------------------------------------------
	l-=np.min(l)
	l/=np.max(l)
	l*=100
	lab[:,:,0]=l

	print ( 'LAB', np.min(lab), np.max(lab) )

	#---------------------------------------------------------------
	# Convert back to RGB and display
	#---------------------------------------------------------------
	im=lab2rgb(lab)*255
	#Image.fromarray(im.astype(np.uint8)).show()
	#input("press enter to continue")
	return im
	
					

if __name__ == "__main__":	
	main()
	print ( 'done.' )
