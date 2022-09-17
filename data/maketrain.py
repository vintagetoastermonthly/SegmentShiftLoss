"""
 * Author: David Knox
 * Created: Sept 2021
 * Purpose: Preprocess images
 * (c) Copyright by Lynker Analytics.
"""
import numpy as np
from numpy.random import random as rnd
import os,sys
from os import listdir
from os.path import isfile
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.exposure import adjust_gamma
import rasterio
import gc

indir = 'MLTraining_Area7_Pre_SansHuman/'
outdir='train/'
inext='tif'
outext='tif'
np.random.seed(123)

prefix='Pre7_'

dim=512

for f in listdir(indir+'images/'):
	if inext == f.split('.')[-1]:
		if os.path.isfile(outdir+'label/0/'+prefix+f.replace(inext,outext)): continue
		imfile=indir+'/images/'+f
		maskfile=indir+'/labels/1/'+f
		if isfile(maskfile):
			print ( imfile )
			try:
				with Image.open(imfile).convert("RGB") as rgb:
					with rasterio.open(maskfile) as mask:
						maskim=mask.read()
						maskim=np.sum(np.swapaxes(np.swapaxes(maskim,0,2),0,1),axis=-1)
						maskim[maskim>0]=255
						maskim[int(rnd()*dim),int(rnd()*dim)]=0
						maskim[int(rnd()*dim),int(rnd()*dim)]=255
						maskim2=np.zeros((dim,dim,3),dtype=np.uint8)
						maskim2[:,:,0]=maskim
						maskim2[:,:,1]=maskim
						maskim2[:,:,2]=maskim

						#---------------------------------------------------------------
						# save to train
						#---------------------------------------------------------------
						with open(outdir+'image/0/'+prefix+f.replace(inext,outext),'wb') as dest:
							rgb.save(dest)
						with open(outdir+'label/0/'+prefix+f.replace(inext,outext),'wb') as dest:
							Image.fromarray(maskim2).save(dest)

				gc.collect()
			except Exception as e:
				print ( 'Exception found:', e )
				print ( 'infile', imfile )
				exc_type, exc_obj, exc_tb = sys.exc_info()
				fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
				print(exc_type, fname, exc_tb.tb_lineno)
				print ( '-----------------------' )
				print ( vars() )
				print ( '-----------------------' )
				#quit()
