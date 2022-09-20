"""
 * Author: David Knox
 * Created: 2020 Nov 14
 * Purpose: ENet based segmentation model
 * (c) Copyright by Lynker Analytics.
"""
from config import *

if modeltype=='ENet':
	from model_ENet import *
elif modeltype=='DeepLab':
	from model_DeepLabV3plus import *
else:
	print ( 'Model Type Unknown', modeltype )
	quit()

if __name__ == "__main__":
	model,age=getmodel(modelfile,shape=(512,512,3))
	print ( model.summary() )
