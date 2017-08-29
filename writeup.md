#**Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/hist_train.jpg "Training Set Histogram"
[image2]: ./writeup_images/hist_valid.jpg "Validation Set Histogram"
[image3]: ./writeup_images/hist_test.jpg "Test Set Histogram"
[image4]: ./German_Sign_Test_Images/sign1_scaled.png "Traffic Sign 1"
[image5]: ./German_Sign_Test_Images/sign2_scaled.png "Traffic Sign 2"
[image6]: ./German_Sign_Test_Images/sign3_scaled.png "Traffic Sign 3"
[image7]: ./German_Sign_Test_Images/sign4_scaled.png "Traffic Sign 4"
[image8]: ./German_Sign_Test_Images/sign6_scaled.png "Traffic Sign 5"


---
### README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data Set Statistics

This Traffic Sign Classifier project used a database of German traffic sign images. The images were all RGB-coded and of size 32x32x3.

There were 43 unique types of traffic signs included in this set, and the size of the training, validation and test sets are as follows:

|   Data Set   |   Number of Images   |
|:------------:|:--------------------:|
|  Training    |     34799            |
|  Validation  |     4410             |
|  Test        |     12360            |


#### 2. Visualization of the dataset.

Here is a histogram visualization of the number of examples of each traffic sign class for each image set:

![Training Set Histogram][image1]
![Validation Set Histogram][image2]
![Test Set Histogram][image3]

### Design and Test a Model Architecture

#### 1. Preprocessing image data

For preprocessing the data, I only normalized the pixel values in the RGB channels to a mean of 0 and standard deviation of 1. I tried converting images to HSV colormapping, however this only degraded performance of the neural network. I kept the RGB channels instead of converting to grayscale as I wanted the neural network to take advantage of the sign colors.


#### 2. Model Architecture
My final model uses 3 convolution layers with Max Pooling after the first and third layers. The output of the final layer is flattened and passed through 4 fully connected hidden layers.

My final model consisted of the following layers:

| Layer         		|     Description		| 
|:---------------------:|:-------------------------------------:| 
| Input         		| 32x32x3 RGB image  		| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 29x29x18 	|
| RELU			|             |
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 11x11x32 	|
| RELU			|             |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x64	|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64	|
| Flatten		|     outputs 1024            |
| Fully connected	|   outputs 500        |
| RELU			|			|
| Fully connected	|   outputs 172        |
| RELU			|			|
| Fully connected	|   outputs 86        |
| RELU			|			|
| Fully connected	|   outputs 43        |
| Softmax		|       	|
											|

#### 3. Training the Model

To train the model, I used the Adam algorithm for optimization with a learning rate of 0.0012 for 8 epochs. The 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


