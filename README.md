# Project 3 - Behavioral Cloning

## udacity Self-Driving Car Engineer Nanodegree
Use Deep Learning to Clone Driving Behavior

### Data Collection

I personally believe the data collection was the most crucial step to success in this project.
I collected data in two ways:

- Centerline Driving: as a baseline, I recorded a few laps of driving with an effort to maintain the car in the center of the lane as much as possible.
- Recovery Driving: driving from either the left or the right side of the road back to the center of the road (not recording the inverser direction, i.e. when the car leaves the center of the road)

From the recordings, I removed about 70% of the images with a steering angle near zero in order to increase the proportion of images with non-zero steering angles. (I found there to be a relatively narrow balance between keeping to many zero-steering images and retaining to few of these images. Keeping to many zero-steering images led to the vehicle driving off-track in sharper corners more easily, while keeping too few zero-steering images led to the vehicle "bouncing" between the left and right markings. The final dataset consistet of around 50,000 images

In order to keep training times / computational requirements low, I decided against including images from the right and left camera. For the same reason, I downsampled images to half their original resolution (320x160 -> 160x80).

### Training

I split the data in to a training, a validation and a test set according to (.75:.125:.125).
The batch size I used for training is 64.

An Adam optimizer is used for optimization and a means squared error (mse) as loss metric.
I first trained the model using a relatively high learning rate of 0.02 for 5 Epochs.
Then I finetuned the model using a much lower learning rate of 0.0001 for another 5 Epochs.

### Model Architecture

The model that I implemented roughly follows the architecture from the recommended [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia. However, the layer input sizes are adjusted according to the different image resolution of the images fed into the network. Also, I added additional dropout layers after each of the fully connected layers to prevent overfitting.
<img src="https://github.com/johannesdeselaers/CarND-Behavioral-Cloning/blob/master/images/architecture.png" width="600">

Specifically, the layers are configured as follows:

1. Input layer, 3x160x80
2. Normalization layer, 3x160x80
3. 5x5 Convolution, 24 filters, followed by batch normaization & activation
4. 5x5 Convolution, 36 filters, followed by batch normaization & activation
5. 5x5 Convolution, 48 filters, followed by batch normaization, activation & dropout
6. 3x3 Convolution, 64 filters, followed by batch normaization & activation
7. 3x3 Convolution, 64 filters, followed by batch normaization, activation & dropout
8. Fully Connected layer with 1164 units, followed by batch normaization, activation & dropout
9. Fully Connected layer with 100 units, followed by batch normaization, activation & dropout
10. Fully Connected layer with 50 units, followed by batch normaization, activation & dropout
11. Fully Connected layer with 10 units, followed by batch normaization, activation & dropout
12. Output layer, one single unit to output the steering angle

All Convlutional layers use valid padding. Rectilinear Units (relu) are used as activations.

###  Result

The final model can steer the car around the first track successfully, see this [video](https://youtu.be/SET69LtT2f8)
