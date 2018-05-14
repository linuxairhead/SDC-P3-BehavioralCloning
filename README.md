# 1. Introduction
The purpose of this project is to build and learn a deep neural network that can mimic the behavior of humans driving a car. Obtained data by running Udaicty Autonomous Simulator, designed Neural Network architecture to training by the obtained data and drive autonomously with Udacity Autonomous Simulator. 

# 2. Data Collection and Simulation Environment
## Data Training Computer 
* Python
* Tesnforflow
* Keras
## Udacity Autonomous Simulator
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

# 3. Data Collection
## Collect the data using simulator
* Run the Udacity Autonomous Simulator for proper version
* After choseing monitor resolution, select play
* Select Training Mode
* Select Record and When file explore pop up, browse output folder to save the file.
* Click Record and Collect the data while driving around track. 

## the data collection method
* Recommand using joystic 
* When circle around track, try to keep on center.
* For the extra data collection, circle around track twice
* For the different data collection, drive around counter clockwise 

# 4. Network Architecture
## Preprocess
  * Normalize the data included the lambda layer with model 
    ```python
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))
    ```
  * Cropping unnecessary image from data with model
    ```python  
    Cropping2D(cropping=((70,25),(0,0)))
    ```
  
## implemented with NVIDIA  
I chosed the NVIDIA's End to End Learning for Self-Driving Cars [paper](https://arxiv.org/pdf/1604.07316v1.pdf). as model as my base model

| Layer (type)                    | Output Shape        | Param # |
|---------------------------------|---------------------|---------|
| lambda_1 (Lambda)               | (None, 160, 320, 3) |  0      |
| cropping2d_1 (Cropping2D)       | (None, 65, 320, 3)  |  0      |
| convolution2d_1 (Convolution2D) | (None, 31, 158, 24) |  1824   |     
| convolution2d_2 (Convolution2D) | (None, 14, 77, 36)  |  21636  |     
| convolution2d_3 (Convolution2D) | (None, 5, 37, 48)   |  43248  |     
| convolution2d_4 (Convolution2D) | (None, 3, 35, 64)   |  27712  |     
| convolution2d_5 (Convolution2D) | (None, 1, 33, 64)   |  36928  |     
| flatten_1 (Flatten)             | (None, 2112)        |  0      |     
| dense_1 (Dense)                 | (None, 100)         |  211300 |     
| dense_2 (Dense)                 | (None, 50)          |  5050   |     
| dense_3 (Dense)                 | (None, 10)          |  510    |     
| dense_4 (Dense)                 | (None, 1)           |  11     |     


# 5. Training
## Data Description
The dataset consists of 8036 rows. Since each row contains images corresponding to three cameras on the left, right, and center, a total of 24,108 images exist.

But, training data is very unbalanced. The 1st track contains a lot of shallow turns and straight road segments. So, the majority of the dataset's steering angles are zeros. Huge number of zeros is definitely going to bias our model towards predicting zeros.

![Steering Histogram](./img/steering_hist.png)

## Data Augmentation
Augmentation refers to the process of generating new training data from a smaller data set. This helps us extract as much information from data as possible.

Since I wanted to proceed with only the given data set if possible, I used some data augmentation techniques to generate new learning data.

### Randomly choosing camera
During the training, the simulator captures data from left, center, and right cameras. Using images taken from left and right cameras to simulate and recover when the car drifts off the center of the road. My approach was to add/substract a static offset from the angle when choosing the left/right camera.

Left | Center | Right
-----|--------|------
![left](./img/left.png) | ![center](./img/center.png) | ![right](./img/right.png)

### Random shear
I applied random shear operation. The image is sheared horizontally to simulate a bending road. The pixels at the bottom of the image were held fixed while the top row was moved randomly to the left or right. The steering angle was changed proportionally to the shearing angle. However, I choose images with 0.9 probability for the random shearing process. I kept 10% of original images in order to help the car to navigate in the training track.

Before | After
-------|-------
![before](./img/before_shear.png) | ![after](./img/after_shear.png)

### Random flip
Each image was randomly horizontally flipped and negate the steering angle with equal probability. I think, this will have the effect of evenly turning left and right.

Before | After
-------|-------
![before](./img/before_flip.png) | ![after](./img/after_flip.png)

### Random gamma correction
Chaging brightness to simulate differnt lighting conditions. Random gamma correction is used as an alternative method changing the brightness of training images.

Before | After
-------|-------
![before](./img/before_gamma.png) | ![after](./img/after_gamma.png)

### Random shift vertically
The roads on the second track have hills and downhill, and the car often jumps while driving. To simulate such a road situation, I shifted the image vertically randomly. This work was applied after image preprocessing.

Before | After
-------|-------
![before](./img/before_bumpy.png) | ![after](./img/after_bumpy.png)

## Data Generators
In this training, I used a generator, which randomly samples the set of images from csv file. As mentioned earlier, because there is a lot of data with a steering angle of 0, I removed this bias by randomly extracting the data. These extracted images are transformed using the augmentation techniques discussed above and then fed to the model.

![Steering Histogram](./img/train_steering_hist.png)

I also used a generator for validation. This generator receives data for validation and returns the corresponding center camera image and steering angle. The validation data was created by leaving only 10% of the row with zero steering angle in the training data.

## Training parameters
* Adam optimizer with a learning rate of 0.01
  * One of the advantages of Batch Normalization is that it can achieve high learning rate.
* 128 batch size
* 5 training epochs
  * Because I used my desktop for training, I tried to use as many ways as possible to reduce my training time.

### Autonomous mode
  * Set up your development environment with the environment.yml
  * Run the server : `python drive.py model.json`
  * You should see the car move around
  
# 6. Drive Autonomously
## Drive with record
Run drive.py using the saved model.
```python
python drive.py model.h5 run1
```
* It receives the image of the central camera from the simulator, preprocesses it, and predicts the steering angle.
* Returns the predicted angle and throttle again.
* The throttle adjusted to 0.5 to climb the hill of track 2.

In my desktop environment, when the simulator was run with high graphics quality, the steering angle prediction slowed down and the car moved around occasionally. So when I run the simulator, I choose the graphics quality as Fastest.

Track 1 | Track 2
--------|--------
![track1](./img/track_one.gif) | ![track2](./img/track_two.gif)

## Generated Video
```python
python video.py run1
```

# 7. Conclusions
This was pretty diff
It was a fun but difficult project. The concept of data augmentation was not difficult because of previous project experience, and the use of generators was quickly becoming familiar. The nvidia model worked better than I expected, but it was pretty hard to get there a bit further. I have repeated various experiments and have reviewed several papers, but it was not as easy as I thought to improve the results. Aboeve all, I was embarrassed that the smaller val_loss did not seem to guarantee a smoother ride.