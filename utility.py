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

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess( ):

    print( " Create sequential Model \n" )	
    model = Sequential()	
	
	# normalized the data by dividing each element by 255 which is the maximum value of an image pixel
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(INPUT_SHAPE)))
    
    # cropping the image to remove unnecessary portion of image.
	# Crop 70 pixels from the top of the image and 25 from the bottom
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(INPUT_SHAPE)))		
	
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
	
	# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
	
	 # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
	
	# Add three fully connected layers (depth 100, 50, 10), than activation 
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

	
def get_Model( arch ):

    """
	disable retraining model using saved model
	
    for fname in os.listdir('.'):	
        if fname.endswith('.h5'):
            #print( " Checking", fname )
            if fnmatch.fnmatch(fname, 'model.h5'):
                print( "**", fname, "loading ... ..." )               
                fname = os.path.realpath(os.path.join('.',fname))	
                model = load_model(fname)
                return model	            
    """
    if arch is 1:
        print(" Training with LeNet Modeling .. ")
        return get_LeNET_Model()	 
    else:
        print(" Training with NVIDIA Modeling .. ")	
        return get_NVDIA_Model()
	

def random_translate(image, steering_angle):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = 100 * np.random.uniform() - 50
    trans_y = 20 * np.random.uniform() - 10
    steering_angle += trans_x * 0.01
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle
	
"""
def random_translate(image,steer,trans_range):
    
    Randomly shift the image virtially and horizontally (translation).
    
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 45*np.random.uniform()- 45/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(IMAGE_WIDTH,IMAGE_HEIGHT))
    
    return image_tr,steer_ang	
"""

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH ]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

	
def random_brighness( image ):    
    """
    Randomly apply brightness to the image 
    ---
	
    newImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    newImage[:,:,2] = newImage[:,:,2] * (.25 + np.random.uniform())
	
    return cv2.cvtColor(newImage, cv2.COLOR_HSV2RGB)	
	"""
    newImage = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    newImage = np.array(newImage, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    newImage[:,:,2] = newImage[:,:,2]*random_bright
    newImage[:,:,2][newImage[:,:,2]>255]  = 255
    newImage = np.array(newImage, dtype = np.uint8)
    newImage = cv2.cvtColor(newImage,cv2.COLOR_HSV2RGB)
    return newImage


def random_flip(image, measurement):
    """
    Randomly flipt the image horizontally 
	and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        measurement = -measurement
    return image, measurement

	
def get_augmentedData( images, measurements ):

    # By adding flipping image And Steering Measurements
    # create twice sample	
	
    augImages = []
    augMeasurements = []	

    for image, measurement in zip(images, measurements):
		
        image, measurement = random_flip(image, measurement)		
        image = random_brighness( image )	
        image, measurement = random_translate(image, measurement)   
        image = random_shadow(image)
		
        augImages.append(image)	
        augMeasurements.append(measurement)
		
    return augImages, augMeasurements    

	
def get_images( trainOrValidation, readData ):

    location="./data/"
    images = []
    measurements = []
    correction = 0.2            
	
    for line in readData:
	
        # Validation get the image for center
		# Training get random image from left, center or right camera
        if trainOrValidation is 0:
            image = cv2.imread(location.strip()+line[0].strip())
			
            if image is None:
                print("Image path incorrect: ", line[i].split('/')[-1])
                continue
				
            measurement = float(line[3])
            images.append(image)	
            measurements.append(measurement)			

        else :			
            # get the image for right, center, and left side camera for each line. 
            for i in range(3):
			
                image = cv2.imread(location.strip()+line[i].strip())

		        #Check the image has been successfully pick up.
		        # if not, skip adding these rows in the for loop
                if image is None:
                    print("Image path incorrect: ", line[i].split('/')[-1])
                    continue

                measurement = float(line[3])
		
                if i is 0:         # center 
                    measurement = measurement			
                elif i is 1:       # left camera
                    measurement += correction
                else:              # right camera
                    measurement -= correction
			
                images.append(image)	
                measurements.append(measurement)
			
    return images, measurements
	

def get_Generator( trainOrValidation, imagesSample, batch_size ):

    num_samples = len(imagesSample)
	
    while 1: # Loop forever so the generator never terminates
		
        for offset in range(0, num_samples, batch_size):
            batch_samples = imagesSample[offset:offset+batch_size]

            images, measurements = get_images( trainOrValidation, batch_samples )
			
			# random 
            if trainOrValidation :
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
