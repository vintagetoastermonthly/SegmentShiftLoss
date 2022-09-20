#---------------------------------------
# image parameters
#---------------------------------------
height=512
width=512
#height=256
#width=256
channels=3
o_channels=1

#---------------------------------------
# model parameters
#---------------------------------------
batch_size=4
#batch_size=16
#batch_size=32
#samples_per_epoch=256
samples_per_epoch=1024
#samples_per_epoch=120
#validation_samples_per_epoch=1024
validation_samples_per_epoch=440
#validation_samples_per_epoch=60
initial_epochs=10
epochs=400
earlystop_patience=399
earlystop_patience_finetune=79
lr_patience=51
workers=1

#modeltype='ENet'
modeltype='DeepLab'

gpu_memory_limit_MB=1024*20

#---------------------------------------
# file locations for full segmentation model
#---------------------------------------
modelfile='models/{}/model.h5'.format(modeltype)
auto_model_file='models/auto.h5'
publishfile='models/ModelZoo/Buildings_AK2017_0075_512_512.h5'
csvfile='logs/epochstats.csv'

train_image_dir='data/train/image/'
train_label_dir='data/train/label/'
valid_image_dir='data/valid/image/'
valid_label_dir='data/valid/label/'
#fine_valid_image_dir='data/Fine_Valid/image/'
#fine_valid_label_dir='data/Fine_Valid/label/'

#valid_image_dir=fine_valid_image_dir
#valid_label_dir=fine_valid_label_dir

