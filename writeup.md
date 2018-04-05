# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The classes are distributed imbalanced from data set.

### Design and Test a Model Architecture

#### 1. Here are how I preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because exposure and lightning varies, some signs are overrexposed, some signs are very dark. And the dimensionality of a image will be reduced 1 depth channel, with data reduction of `66%`.

As a last step, I normalized the image data because the image data should be normalized so that the data has mean zero and equal variance minimally.

#### 2. Here I describe what the final model architecture looks like including model type, layers, layer sizes, connectivity, etc. including a diagram table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400 inputs, 120 outputs	|
| RELU					|												|
| Fully connected		| 120 inputs, 84 outputs	|
| RELU					|												|
| Fully connected		| 84 inputs, 43 outputs	|


#### 3. Here is about how I trained your model.

To train the model, I used following hyperparameters.

- `LEARNING_RATE = 0.001`
- `EPOCHS = 10`
- `BATCH_SIZE = 64`

#### 4. The final solution list here getting the validation set accuracy is over 0.93. 

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.979
* test set accuracy of 0.906


About the architecture was chosen:
* I started with LeNet, and it works.
* I preprocessed the image data, with LeNet it works.
* I adjusted the batch size and EPOCH number to fit my system, to archieve highest accuracy and save training time.

### Test a Model on New Images

#### I Choose five German traffic signs found on the web and test with program well.

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of `0.906`.

