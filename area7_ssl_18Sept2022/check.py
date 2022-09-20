"""
 * Author: David Knox
 * Created: Apr 2022
 * check that all images and labels are matched
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

errorfile=[]
for indir in ['train/','valid/']:
	print ( indir )
	for subdir in ['image/','label/']:
		print ( indir, subdir )
		for f in listdir(indir+subdir+'0/'):
			imfile=indir+'/image/0/'+f
			lbfile=indir+'/label/0/'+f
			if not isfile(lbfile):
				errorfile.append(imfile)
			if not isfile(imfile):
				errorfile.append(lbfile)

for ef in errorfile:
	print ( 'del', ef.replace('/','\\') )
