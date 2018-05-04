import os
import csv
import cv2
import sklearn
import fnmatch
import utility
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split


def preprocess( ):

    print( " Create sequential Model \n" )	
    model = Sequential()	
	
	# normalized the data by dividing each element by 255 which is the maximum value of an image pixel
    # model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
    
    # cropping the image to remove unnecessary portion of image.
	# Crop 70 pixels from the top of the image and 25 from the bottom
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))		
	
    return model

	
def get_LeNET_Model():

    model = preprocess()
    ### training with LeNet Architecture. ###
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.summary()
    return model

	
def get_NVDIA_Model():

    model = preprocess()
    ### training with NVIDIA Architecture. ###
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

	
def get_Model( arch ):

    for fname in os.listdir('.'):	
        if fname.endswith('.h5'):
            #print( " Checking", fname )
            if fnmatch.fnmatch(fname, 'model.h5'):
                print( "**", fname, "loading ... ..." )               
                fname = os.path.realpath(os.path.join('.',fname))	
                model = load_model(fname)
                return model	            
    
    if arch is 1:
        print(" Training with LeNet Modeling .. ")
        return get_LeNET_Model()	 
    else:
        print(" Training with NVIDIA Modeling .. ")	
        return get_NVDIA_Model()
	
			
def get_augmentedData( images, measurements ):

    # By adding flipping image And Steering Measurements
    # create twice sample	
    augmented_images = []
    augmented_measurements = []

    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*(-1.0))
	
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
	
    return X_train, y_train

	
def get_images( readData ):

    images = []
    measurements = []
    correction = 0.2
	
    for line in readData:
	
        # get the image for right, center, and left side camera for each line. 
        for i in range(3):
		
            #image = cv2.imread(line[i].split('/')[-1])
            #print(line[i].split('/')[-1])
            location="./data/"
            image = cv2.imread(location.strip()+line[i].strip())
            #print(location.strip()+line[i].strip())	
            #print(os.getcwd())
			# Check the image has been successfully pick up.
			# if not, skip adding these rows in the for loop
            if image is None:
                print("Image path incorrect: ", line[i].split('/')[-1])
                continue

            measurement = float(line[3])
		
            if i is 1:         # left camera
                measurement += correction			
            elif i is 2:       # right camera
                measurement -= correction
            else:
                measurement = measurement
			
            images.append(image)	
            measurements.append(measurement)
    #print(image.shape) 	
    #input()	
    #plt.figure(figsize=(25,25))
    #plt.subplot(1, 5, 1)
    #plt.imshow(image[0])
    #plt.show()    
	
            #print( "line", line, "index", i, "measurement appended", measurement)				
    return images, measurements
	

def get_Generator( imagesSample, batch_size ):

    num_samples = len(imagesSample)
	
    while 1: # Loop forever so the generator never terminates
		
        for offset in range(0, num_samples, batch_size):
            batch_samples = imagesSample[offset:offset+batch_size]

            images, measurements = get_images( batch_samples )
			
			# random 
            if np.random.rand() < 0.5:
                images, measurements = get_augmentedData( images, measurements )			
                    		
            yield np.array(images), np.array(measurements)

			
def get_SampleData( csv_file ):
	samples = []
	with open(csv_file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)

	# remove first line from excel file since it doesn't contain the data. 
	del(samples[0])
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	
	return train_samples, validation_samples
