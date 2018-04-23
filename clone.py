import csv
import cv2
import numpy as np

architecture = 2 # 1 for LeNet, 2 for NVIDIA
lines = []
csv_file = '../simulator-self-driving-car/Data2/driving_log.csv'
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# remove first line from excel file since it doesn't contain the data. 
del(lines[0])

images = []
measurements = []
correction = 0.2

for line in lines:

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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()

### preprocess the image ###

# normalized the data by dividing each element by 255 which is the maximum value of an image pixel
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# cropping the image to remove unnecessary portion of image.
model.add(Cropping2D(cropping=((70,25),(0,0))))

if architecture is 1:

    ### training with LeNet Architecture. ###
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

elif architecture is 2:

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
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')