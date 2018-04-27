import csv
import cv2
import utility
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def preprocess( ):
    model = Sequential()
	
	# normalized the data by dividing each element by 255 which is the maximum value of an image pixel
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

    # cropping the image to remove unnecessary portion of image.
    model.add(Cropping2D(cropping=((70,25),(0,0))))		
	
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
	
    return model

	
def get_model( arch ):
    
	if arch is 1:
	    return get_LeNET_Model()
	   
	else :
	    return get_NVDIA_Model()
	
	
def get_augmentedData( data ):

    images = data[0]
    measurements = data[1]
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

			image = cv2.imread(line[i].split('/')[-1])
	
			# Check the image has been successfully pick up.
			# if not, skip adding these rows in the for loop
			if image is None:
				print("Image path incorrect: ", line[i].split('/')[-1])
				continue
				
			images.append(image)
			measurement = float(line[3])		
		
			if i is 0:   # center camera
				measurements.append(measurement)
			
			elif i is 1: # left camera
				measurements.append(measurement + correction)
			
			else:        # right camera
				measurements.append(measurement - correction)
				
	return images, measurements
	
			
def read_csv( csv_file ):
	lines = []
	with open(csv_file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	# remove first line from excel file since it doesn't contain the data. 
	del(lines[0])
	
	return lines

	
def get_data( csv_file ):

    readData = read_csv( csv_file )
    images = get_images( readData )
    images = get_augmentedData( images )
	
    return images