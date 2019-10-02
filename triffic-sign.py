import numpy as np
import tensorflow as tf
import time
import pandas as pd 
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.en
semble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import keras
import os
import skimage.data
import skimage.transform
import matplotlib
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from sklearn.utils import shuffle
import glob
from skimage import io, color, exposure, transform
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import cv2
from PIL import Image
from keras.layers.normalization import BatchNormalization

NUM_CLASSES = 119
IMG_SIZE = 40
csvfile=pd.read_csv('training.csv')
Label = csvfile['class']
K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')
root_dir = 'img/'
imgs = []
Labels = []

def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(3, IMG_SIZE, IMG_SIZE)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax')) 
    model = Sequential()
    return model
  
model = cnn_model()
model.summary()


def dataprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = color.hsv2rgb(hsv)
    img = np.array(img)
    x1 = csvfile['x1'][i]
    y1 = csvfile['y1'][i]
    x2 = csvfile['x2'][i]
    y2 = csvfile['y2'][i]
    h = (y2-y1)
    w = (x2-x1)
    img = img[y1:y1+h,x1:x1+w]
    return img

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))
  
data_gen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              fill_mode='nearest',
                              data_format='channels_last')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
  
print("processing...")
all_img_name = csvfile['img']
samplenum = len(all_img_name)
for i in range(samplenum):
    img_name = all_img_name[i]
    img_path = (root_dir+img_name)
    img = io.imread(img_path)
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img = color.hsv2rgb(hsv)
    img = dataprocess(io.imread(img_path))
    img = np.array(img)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), 3)
    img = np.rollaxis(img, -1) 
    imgs.append(img)
X = np.array(imgs, dtype='float64')
#Labels = tf.one_hot(Label, NUM_CLASSES)
print("finish!")

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
history = LossHistory()

model.fit(X, Label,
          steps_per_epoch = 200,
          epochs=10,
          validation_steps=True,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     history,
                   ModelCheckpoint('model.h5', save_best_only=True)]
          )
history.loss_plot('epoch')