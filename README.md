# **Behavioral Cloning** 

## Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Image from center camera"
[image2]: ./examples/center_curve_left.jpg "Curve left - center camera"
[image3]: ./examples/center_curve_right.jpg "Curve right - center camera"
[image4]: ./examples/left_curve_left.jpg "Curve left - left camera"
[image5]: ./examples/right_curve_left.jpg "Curve left - right camera"
[image6]: ./examples/center.jpg "Original image"
[image7]: ./examples/center_cropped.jpg "Cropped image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

I did a lot of trials, and arrived at three different network architectures that completed the 1st circuit. I include model2.py and model2.h5 as well as model3.py and model3.h5 for those other networks, and did a comparison of results from the three of them.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

the other networks can be run using model2.h5 or model3.h5.


#### 3. Submission code is usable and readable

The model.py, model2.py and model3.py files contains the code for training and saving the convolution neural networks. The file shows the pipeline I used for training and validating the models, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In all models, I cropped the images using keras' Cropping2D as the first layer:

```python
model.add(Cropping2D(cropping=((60, 23), (0, 0)), input_shape=(160, 320, 3)))
```

then follows a normalization layer using a Lambda:

```python
model.add(Lambda(lambda x: x / 127.5 - 1.))
```

After the normalization comes the model itself. In all cases I used RELU layers to introduce nonlinearities, and a dropout to avoid overfitting.

`Note`: timing and loss numbers can be different if the model is run again, even on the same machine. To work on this project I used my personal notebook which has an NVidia GTX 960M graphic card with 2GB memory.

##### Model 1
Model1 consists of 5 main layers:
* Convolutional , kernel 5x5, depth 10, followed by a RELU activation layer and a Max Pooling 2x2 layer.
* Convolutional, kernel 5x5, depth 6, followed by RELU activation and Max Pooling 2x2
* Dense layer with 120 neurons
  * dropout with 0.5 probability
* Dense layer with 84 neurons
  * dropout with 0.4 probability
* Dense layer with 1 neuron for the output

Trained by 5 epochs took 909 seconds
final loss: 0.0348 - validation loss: 0.0490


##### Model 2
This model is the one NVIDIA used according to the class video:
* Convolutional, kernel 5x5, depth 24. RELU activation, Max Pooling 2x2
* Convolutional, kernel 5x5, depth 36. RELU activation, Max Pooling 2x2
* Convolutional, kernel 5x5, depth 48. RELU activation, Max Pooling 2x2
* Convolutional, kernel 3x3, depth 64. RELU activation
* Convolutional, kernel 3x3, depth 64. RELU activation
* Dense, 100 neurons
  * dropout 0.5 prob
* Dense, 50 neurons
* Dense, 10 neurons
* Dense, 1 neuron as output

5 epochs training time: 944 seconds
final loss: 0.0282 - val_loss: 0.0301


##### Model 3
This model is a simplification of the NVIDIA one, as I felt that it was needlessly complex. I took out final two convolutional layers:
* Convolutional, kernel 5x5, depth 24. RELU activation, Max Pooling 2x2
* Convolutional, kernel 5x5, depth 36. RELU activation, Max Pooling 2x2
* Convolutional, kernel 5x5, depth 48. RELU activation, Max Pooling 2x2
* Dense, 100 neurons
  * dropout 0.5 prob
* Dense, 50 neurons
* Dense, 10 neurons
* Dense, 1 neuron as output

5 epochs training time: 
final loss: 0.0369 - validation loss: 0.0393


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 71).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).


#### 4. Appropriate training data

Main training data was provided by Udacity. This data was augmented by manual driving, recovering from both sides of the road, and closed curves.
For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a very simple model: one convolutional layer and two dense layers. This took the car to the first curve, where it kept going straight. Tried to compensate augmenting the data with a couple of manual driven laps, but the results were only marginally better.

I then added another convolutional and another dense layers to get to `model 1`. This model behaved pretty well, but the car consistently missed the closed curve after the bridge and ended stranded in the field outside the road. So I augmented the data recording several turns specifically and some recovery maneuvers from both sides. Finally, the model was able to complete the circuit but sometimes the car leaned to the left or right of the road. Now I augmented the data by using the recorded images for left and right cameras, with a small correction to the angle registered to compensate. I chose randomly between left, center or right images during training, and trained for more batches to try to cover all images. This setup was able to do the circuit pretty well.
I still noted much 'flicker' on the angles though: the car seemed to be twitching quickly, although that could not take it away from the road.
In an attempt to reduce the twitching, I modified drive.py to output the average of N predictions instead of each one 'raw'. It helped somehow, but the car started oscillating inside the road. So I left it as it was. The code is there in drive.py anyway (lines 70-74), just have to change the number of angles to consider for the averaging, setting it in line 21.

Now that I had reached the objective, I tried with NVIDIA network architecture to see if the added complexity payed off. I noted some less twitching, but the overall result did not change much. This architecture is in model2.py.

Finally, I removed the last two convolutional layers from the NVIDIA architecture to see if the lack of feature discovery had any effect. Code is in model3.py. I also saw a reduced flicker comparing with model1, but in a section of the track the car was consistently brougth to the left, touching the border but without leaving the road. The time it took to train was not much different from the full NVIDIA arch, so that would be my choice for this exercise.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). 

I found that my first model had a low mean squared error on the training set and higher mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model adding a dropout with probability of survival of 50% between the first two dense layers. This helped, but still validation error was higher than training, so I added another dropout in the next dense layer.

The vehicle is able to drive autonomously around the track without leaving the road with any of the three architectures.


#### 2. Final Model Architecture

Of the three architectures tested, all were able to complete the track without steering out of the road, but the third stepped over the border at one place so I think the first two are better in this particular scenario.
Of these two, I would stay with the simpler: the first one. 
As an added test, I ran the three nets at a higher speed.
* At 20mph, both the first and the second networks could complete the lap, both oscillating much more than in low speed.
* At 30mph, the oscillations got worse but the first network was able to stay on course while the second one went out of the road and ended in the water.

So, clear winner for me: the first network architecture.

BUT... things changed when I consider generalizing with track 2 data. I tested adding a couple of laps on track 2 to the data to see how well the network generalized. It turned out that net 1 was not able to complete even track 1 anymore, while net 2 did very well both on track 1 and track 2. So, it seems the extra layers did have a good effect if there is enough data. The car did almost all track 2, just failing on last turns, the ones in the shadow. I guess adding some more training on those shadow curves will make it possible for the net to complete both tracks.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a couple laps on track one. Here is an example image of center lane driving:

![alt text][image1]

the program records images from left and right cameras as well, so I added those to the training with a little correction as to the steering angle which is for center camera. Here is how the same spot looks like from all three cameras:

![alt text][image4]![alt text][image2]![alt text][image5]

The first track has a clear bias towards left turns, so the network leaned to the left and on first trials, it didn't complete the right turn that comes after the bridge. When I was about to flip the images to compensate, I realized that the data set that Udacity offered included both types of images so I used that data set instead.

![alt text][image2]![alt text][image3]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the track in case it veered too far.
After that, I augmented some more the data recording only the worst curves so those spots where reinforced. Kind of what we humans do when learning some skill: reinforce the parts we find more challenging.
With this data, networks 1 and 2 were able to complete track 1. I then tried them on track 2, and the results were awful: the car barely passed the first curve. So I recorded some laps on track 2 to add the data to the training/validation sets.
Testing with the fully augmented data, I see that network 1 is able to complete track 1 but not track 2, though it was better this time.
Network 2, however, was able to complete track 1 and almost all of track 2; it only failed on last two turns with the road on the shadow of the hills. I took control to get the car back on the road and it completed the track. I think it needs to reinforce those difficult spots by repeating them a couple more times, or process the images with some transform that eliminates the effect of the shadows.

All images are cropped to leave only the relevant part with the road:

![alt text][image6]
![alt text][image7]

Each time, the training process included random shuffle of the data set and the taking 20% of the data for validation. 
I ended up using 5 epochs as it offered a good balance between training time and results. I tried more epochs and the loss reduced much more slowly -if at all- and with less epochs the network performed worst.

I saw that the deeper architecture -NVIDIA's one- performed better when it had more training data, even generalizing both tracks to the point to be able to run on both. While the simpler network #1 performed good with less data on first track, but it was confused when I added data from track 2. An early conclusion would be that to be used in a real environment, where everything changes all the time, the network has to be deep enough to store knowledge from a vast array of different situations. And train it with a LOT of data. It could also help to use some preprocessing to the images to eliminate lighting effects.
