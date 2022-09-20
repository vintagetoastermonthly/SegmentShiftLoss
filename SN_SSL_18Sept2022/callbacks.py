from config import *
from tensorflow.keras.callbacks import ModelCheckpoint #(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
from tensorflow.keras.callbacks import EarlyStopping #(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
from tensorflow.keras.callbacks import CSVLogger #(filename, separator=',', append=False)
from tensorflow.keras.callbacks import ReduceLROnPlateau #(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

#monitor='val_binary_io_u'
monitor='val_loss'
#monitor='val_BIoU'

MySaver=ModelCheckpoint(modelfile, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
MyCSV=CSVLogger(csvfile, separator=',', append=True)
MyStopper=EarlyStopping(monitor=monitor, min_delta=0, patience=earlystop_patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
MyStopperFinetune=EarlyStopping(monitor=monitor, min_delta=0, patience=earlystop_patience_finetune, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
MyLR=ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=lr_patience, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[MySaver,MyCSV,MyStopper,MyLR]
finetune_callbacks=[MySaver,MyCSV,MyStopperFinetune]
