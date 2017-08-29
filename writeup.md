#**Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/hist_train.png "Training Set Histogram"
[image2]: ./writeup_images/hist_valid.png "Validation Set Histogram"
[image3]: ./writeup_images/hist_test.png "Test Set Histogram"
[image4]: ./writeup_images/preprocessed_images.png "Preprocessed Images"
[image11]: ./German_Sign_Test_Images/sign1_scaled.jpg "Traffic Sign 1"
[image12]: ./German_Sign_Test_Images/sign2_scaled.jpg "Traffic Sign 2"
[image13]: ./German_Sign_Test_Images/sign3_scaled.jpg "Traffic Sign 3"
[image14]: ./German_Sign_Test_Images/sign4_scaled.jpg "Traffic Sign 4"
[image15]: ./German_Sign_Test_Images/sign6_scaled.jpg "Traffic Sign 5"
[image30]: ./writeup_images/sign30_grayscale.png "30 Grayscale"
[image31]: ./writeup_images/sign30_layer1.png "30 Layer 1"
[image32]: ./writeup_images/sign30_layer2.png "30 Layer 2"
[image80]: ./writeup_images/sign80_grayscale.png "80 Grayscale"
[image81]: ./writeup_images/sign80_layer1.png "80 Layer 1"
[image82]: ./writeup_images/sign80_layer2.png "80 Layer 2"

#### Project Code

[Link to iPython Notebook](https://github.com/charlestytler/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

In addition, the Jupyter notebook shows an example of each unique class of traffic sign from the training set of images.

### Design and Test a Model Architecture

#### 1. Preprocessing image data

For preprocessing the data, I converted the RGB images to grayscale, and equalized their brightness/contrast with OpenCV's equalize histogram function. I then normalized the pixel values to a range of [-1,1].

I had originally tried the same model architecture with the 3 RGB channels as inputs (after adjusting the pixel range to [-1,1]), however it achieved roughly the same accuracy (within ~1%) on the validation set. So I decided to stick with the single channel input as it is less data being processed.

An example of each class of traffic sign after the grayscale and histogram equalization pre-processing is shown below.

![Pre-Processed Image Examples][image4]

#### 2. Model Architecture
My final model uses 3 convolution layers with Max Pooling after the first and third layers. The output of the final layer is flattened and passed through 4 fully connected layers, with a dropout for regularization.

My final model consisted of the following layers:

| Layer         		|     Description		| 
|:---------------------:|:-------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  		| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU			|             |
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 11x11x32 	|
| RELU			|             |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 9x9x64	|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64	|
| RELU			|             |
| Flatten		|     outputs 1024            |
| Fully connected	|   outputs 500        |
| RELU			|			|
| Fully connected	|   outputs 172        |
| RELU			|			|
| Fully connected	|   outputs 86        |
| RELU			|			|
| Dropout		|   Keeps 25% of Activations  |
| Fully connected	|   outputs 43        |
| Softmax		|       	|


#### 3. Training the Model

To train the model, I used the Adam algorithm for optimization with a learning rate of 0.001 for 8 epochs. The validation accuracy seemed to start oscillating after 7 or 8 epochs, so I just set had the training stop there.

#### 4. Tuning the Model

I started with the LeNet model as a baseline to make sure everything was working as expected and generating somewhat adequate accuracy with percentages in the low 90s. From there I tried experimenting with different adjustments to the model layers to see how it would affect the validation set accuracy.

By adding and removing fully connected layers and convolution layers I worked with the assumption that a deeper model would help. However I didn't go much past three total convolution layers as the images are already fairly low resolution. The third convolution layer I deemed necessary though in order for the model to differentiate higher level shapes. By including a set of decreasing size fully connected layers before the final logits output I found the accuracy also increased significantly.

I attempted including information from the second and third layers together by concatenating their outputs after flattening before passing through the fully connected layers. This only muddied the accuracy though, and so I abandoned this idea.

Once I had a layer architecture that seemed to be working well, I tried increasing the number of neurons for different layers to see how effective they would be, manually adjusting the layers until a high accuracy was achieved in training. I tried to decrease neurons and size of the model wherever it didn't hurt accuracy to prevent the model from becoming too slow.

Finally, I investigated dropouts between various layers in the model for regularization. I tried having two dropout locations, but found that the technique was most effective in the last layer before the logits for the classes are determined. Adjusting the keep probability of the dropout at that location from my initial value of 50%, I found 25% produced better results in validation accuracy.

In the end, my final model achieved an accuracy of:
* Validation set accuracy = 97.6%
* Test set accuracy = 94.9%


### Test a Model on New Images

#### 1. Testing with five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![Sign1][image11] ![Sign2][image12] ![Sign3][image13] 
![Sign4][image14] ![Sign5][image15]

The first two images I chose since the 3 and the 8 have very similar shapes and I wanted to see how well the model would differentiate them. The third sign seemed a simple sign to distinguish as a basic test. The fourth sign has overly bright lighting/coloring in the image and is skewed from the photo angle. The fifth sign also has a slightly skewed circle shape, but I think the oblique arrow direction may make it a difficult sign to classify correctly.


#### 2. Model predictions for the signs
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Sign1][image11] 
87.5%	Speed limit (30km/h)	Occurences in Training Set = 720
10.4%	Speed limit (80km/h)	Occurences in Training Set = 630
0.6%	Speed limit (20km/h)	Occurences in Training Set = 60
0.6%	Speed limit (70km/h)	Occurences in Training Set = 660
0.3%	Speed limit (50km/h)	Occurences in Training Set = 750

The model correctly predicts the 30 km/h sign, and as expected also predicts with some probability it alternatively is a 80 km/h sign.


![Sign2][image12] 
90.1%	Speed limit (60km/h)	Occurences in Training Set = 450
9.9%	Speed limit (80km/h)	Occurences in Training Set = 630
0.0%	Speed limit (30km/h)	Occurences in Training Set = 720
0.0%	Keep right          	Occurences in Training Set = 690
0.0%	Speed limit (100km/h)	Occurences in Training Set = 450

Sign 2 is incorrectly determined to be a 60 km/h sign, however the correct classification of 80 km/h is its second highest probability (only at 10% though).


![Sign3][image13] 
99.9%	No vehicles         	Occurences in Training Set = 210
0.1%	Speed limit (120km/h)	Occurences in Training Set = 450
0.0%	Speed limit (80km/h)	Occurences in Training Set = 630
0.0%	Speed limit (70km/h)	Occurences in Training Set = 660
0.0%	Speed limit (60km/h)	Occurences in Training Set = 450

Sign 3 is pretty much dead on as a No Vehicles sign, as expected for a mostly featureless sign.


![Sign4][image14] 
24.7%	Speed limit (60km/h)	Occurences in Training Set = 450
22.6%	Speed limit (80km/h)	Occurences in Training Set = 630
16.7%	Turn left ahead     	Occurences in Training Set = 120
15.8%	Keep right          	Occurences in Training Set = 690
5.2%	Speed limit (30km/h)	Occurences in Training Set = 720

Sign 4 proves difficult for the model and it doesn't really have a strong prediction for any one label. The correct classification of "Turn Left Ahead" has a 16.7% probability. One thing I noted is that the correct label had one of the fewest occurences in the training set which may have hindered the model.


![Sign5][image15]
53.3%	Roundabout mandatory		Occurences in Training Set = 90
13.0%	End all speed/passing limits	Occurences in Training Set = 60
5.9%	Go straight or left 		Occurences in Training Set = 60
5.5%	Speed limit (30km/h)		Occurences in Training Set = 720
4.6%	End of speed limit (80km/h)	Occurences in Training Set = 150
Sign 5 doesn't even have the correct label in the top 5 probabilities. It really had trouble with this image, but I'm not entirely sure why.



The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This shows a lack of robustness in the model, compared to the accuracy percentages in the 90s for the sets from the source database. Perhaps augmenting the training data set would improve its prediction capability.

I thought that the training set may not have had a variety of different scalings of the sizes. However I tried cropping the test images from the web to more closely match the training set and the accuracy did not improve. Perhaps augmenting the training images with stretched/skewed and/or noisy/blurry versions would improve performance.


### Visualizing the Neural Network

As a quick look at the identification being performed by my model, I created images of the feature maps for the first two convolution layers comparing the 30 km/h and 80 km/h speed limit signs:

30 km/h Speed Limit with Grayscale Preprocessing:

![30_Grayscale][image30]

80 km/h Speed Limit with Grayscale Preprocessing:

![80_Grayscale][image80]

30 km/h Speed Limit output of Layer 1:

![30_Layer1][image31]

80 km/h Speed Limit output of Layer 1:

![80_Layer1][image81]

30 km/h Speed Limit output of Layer 2:

![30_Layer2][image32]

80 km/h Speed Limit output of Layer 2:

![80_Layer2][image82]




