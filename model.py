
# coding: utf-8

# In[ ]:

###############################################################################
# import required packages

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Convolution2D, Flatten,BatchNormalization,SpatialDropout2D,Dropout,Lambda
from keras.optimizers import Adam
import numpy as np
import csv
from random import shuffle
import cv2
import csv
import math
import json


# In[ ]:

###############################################################################

# Some Variables for training and data

fine_tune_model = True
if fine_tune_model:
    learning_rate = 0.000001  
else:
    learning_rate = 0.002  

image_sizeX = 160
image_sizeY = 80
num_channels = 3 

nb_epoch = 1
batch_size = 16


# In[ ]:

###############################################################################

# Auxiliary functions

# Read in the image and resize (half size)
def process_image(filename):
    image = cv2.imread("data/" + filename)
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    return image

# Compute the number of samples per epoch dependent on batch size
def get_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return number divisible by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# In[ ]:

###############################################################################

# Class for the model architecture as in the paper by nvidia
# on end-to-end learning for Self-Driving Cars

class ModelArchitecture(object):
    def __init__(self):
        return
    
    def setup_model(self):
        drop_out = 0.1
        
        model = Sequential()
        
        model.add(Lambda(lambda x: x/127.5 - 1.,                  input_shape=(80, 160, 3),                  output_shape=(80, 160, 3)))
        
        # 5 Convolutional Layers
        # Some with just Batch Normalization
        # Some with Batch Normalization and Spatial Dropout
        model.add(Convolution2D(24, 5, 5, border_mode='valid',subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Convolution2D(36, 5, 5, border_mode='valid',subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Convolution2D(48, 5, 5, border_mode='valid',subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(drop_out))

        model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Convolution2D(64, 3, 3, border_mode='valid',subsample=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(drop_out))

        # Flatten and Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))

        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))

        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))

        model.add(Dense(10,))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))

        model.add(Dense(1))
        
        # Use the Adam Optimizer
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        self.model = model
        return
    
    def get_model(self):
        return self.model


# In[ ]:

###############################################################################

# Get available training data
# and split into training/validation/test sets

with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)
print("Found {} frames.".format(num_frames-1))

X_train = [("", 0.0) for x in range(num_frames-1)]
for i in range(num_frames-1):
    X_train[i] = (driving_log_list[i+1][0].lstrip(), float(driving_log_list[i+1][3]))
    
num_frames = len(X_train)

shuffle(X_train)
num_train_elements = int((num_frames/4.)*3.)
num_validation_elements = int(((num_frames/4.)*1.) / 2.)

X_valid = X_train[num_train_elements:num_train_elements + num_validation_elements]
X_test = X_train[num_train_elements + num_validation_elements:]
X_train = X_train[:num_train_elements]

print("X_train: {} elements.".format(len(X_train)))
print("X_valid: {} elements.".format(len(X_valid)))
print("X_test: {} elements.".format(len(X_test)))


# In[ ]:

###############################################################################

# Generator to provide data to model.fit_generator
def generator_function(data):
    index = 0
    while True:
        images = np.ndarray(shape=(batch_size, image_sizeY, image_sizeX, num_channels), dtype=float)
        angles = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            if index >= len(data):
                index = 0
                # shuffle data every epoch
                shuffle(data)
            image_filename = data[index][0]
            angle = data[index][1]
            final_image = process_image(image_filename)
            final_angle = np.ndarray(shape=(1), dtype=float)
            final_angle[0] = angle
            images[i] = final_image
            angles[i] = final_angle
            index += 1
        yield (images, angles)


# In[ ]:

###############################################################################

# different actions if we want to finetune an existing
# or train a new model from scratch

if fine_tune_model:
    print("Training: fine tuning model")
    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
    
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse',metrics=['mean_absolute_error'])
else:
    print("Training: new model")
    n_model = ModelArchitecture()
    n_model.setup_model()
    model = n_model.get_model()


# In[ ]:

###############################################################################

# Perform training & run test set

print("Epochs : {}".format(nb_epoch))
print("Batch Size : {}".format(batch_size))

if fine_tune_model:
    print("Finetuning with learning rate {}.".format(learning_rate))
else:
    print("Training new model with learning rate {}.".format(learning_rate))

model.fit_generator(
    generator=generator_function(X_train),
    nb_epoch=nb_epoch, 
    max_q_size=10, # max items to be queued, i.e. ready to use, in generator 
    samples_per_epoch=get_samples_per_epoch(len(X_train), batch_size),
    validation_data=generator_function(X_valid),
    nb_val_samples=get_samples_per_epoch(len(X_valid), batch_size),
    verbose=1)

print('Training complete. Now checking test error...')

# Evaluate the accuracy of the model using the test set
score = model.evaluate_generator(
    generator=generator_function(X_test), \
    val_samples=get_samples_per_epoch(len(X_test), batch_size))
print("Test score {}".format(score))


# In[ ]:

###############################################################################

# save model to disk

with open("new_model.json", "w") as text_file:
    text_file.write(model.to_json())
model.save_weights('new_model.h5')

print("Saved model to disk")

