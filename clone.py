import csv
import cv2
import numpy as np

lines = []
with open('../simulator-self-driving-car/Data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del(lines[0])

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    #current_path = '../simulator-self-driving-car/Data2/IMG/' + filename
    current_path = filename
    image = cv2.imread(current_path)
	
	# Check the image has been successfully pick up.
	# if not, skip adding these rows in the for loop
    if image is None:
        print("Image path incorrect: ", current_path)
        continue
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

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

model.save('model.h5')