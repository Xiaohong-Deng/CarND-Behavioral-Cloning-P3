# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./nvidia_net.jpg "Model Visualization"
[image2]: ./center_2017_11_30_06_44_21_848.jpg "Center Image"
[image3]: ./left_2017_11_30_06_54_49_931.jpg "Left Image"
[image4]: ./right_2017_11_30_06_54_59_767.jpg "Right Image"
[image5]: ./elu.jpg "ELU Function"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup report as README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with the following structure

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						|
| Cropping2D            | crop the images to 80x320x3, discarding       |
|                       | irrelevant pixels                             |
| Lambda                | normalize the inputs with function:           |
|                       | x/255.0 - 0.5                                 |
| Convolution 5x5     	| 2x2 stride, no padding, outputs 38x158x24 	|
|                       | previous layer to zero                        |
| ELU					| activation function							|
| Dropout               | randomly set some activation maps from        |
| Convolution 5x5	    | 2x2 stride, no padding, outputs 17x77x36      |							
| ELU                   |                                               |
| Convolution 5x5       | 2x2 stride, no padding, outputs 7x37x48       |
| ELU                   |                                               |
| Convolution 3x3       | 1x1 strides, no padding, outputs 5x35x64      |
| ELU                   |                                               |
| Convolution 3x3       | 1x1 strides, no padding, outputs 3x33x64      |
| ELU                   |                                               |
| Flatten               | outputs 6336                                  |
| Fully connected		| output 100        							|
| Fully connected       | output 50                                     |
| Fully connected       | output 10                                     |
| Fully connected       | output 1                                      |        			


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 77). Drop rate 0.8 worked pretty well.

The model was trained and validated on different data sets to ensure that the model was not overfitting (data splitted in line 19, validated in line 94 to line 97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected data by driving in the center of the road. 

Extracted images from left, center and right camera with additional steering angles added to my driving angles for left and right camera images such that they can be treated as if they were from the center camera.

I also flipped all the images. Steering angles had to be flipped, too.

So I have 6 times of the images as I have from the center camera.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I just employed the NVidia network recommended in the instructions. It worked very well. I just added a Dropout layer. And I used ELU as activation function because I was worried RELU would kill some of the neurons. And it's closer to zero centered output.

![alt text][image5]

The training and validation steps were surprisingly smooth for me. I only trained for 3 epochs. The training and validation loss were pretty good after the first epoch. They did decrease in the 2nd and 3rd epoch, but only slightly.

Auto driving in the simulator was also smooth. Car was driven steadily. Almost always stays in the center of the road. Perhaps a little bit to the right sometimes. Must be my driving data.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-89) consisted of a convolution neural network with 5 convolution layers and 3 fully connected layers. Detailed structure was documented in the previous sections.

Here is a visualization of the architecture from the NVidia paper. Note some layers might have different sizes from mine, but the overall structure is the same

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded over two laps on track one using center lane driving. Here are example images of center lane driving from center, left and right camera:

![alt text][image2]
![alt text][image3]
![alt text][image4]


After the collection process, I had 6761 lines of data points. I then preprocessed this data by extracting left and right camera images, flipping all the images. Effectively retrieved over 40k images


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 3 epochs but I suspect 1 epoch could have done the job because in the 2nd and 3rd epoch the loss was only improved slightly. In all 3 epochs the validation loss was always better than the training loss. NVidia net is powerful!

I used an adam optimizer so that manually training the learning rate wasn't necessary.
