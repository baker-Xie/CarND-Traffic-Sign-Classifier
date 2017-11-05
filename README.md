# Traffic Sign Classifier

In this project, a CNN is used to classify traffic signs [German Traffic Sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)).

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

I used the numpy/pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 

![image](https://github.com/baker-Xie/CarND-Traffic-Sign-Classifier/raw/master/sources/1.png)

![image](https://github.com/baker-Xie/CarND-Traffic-Sign-Classifier/raw/master/sources/3.png)

It is a bar chart showing how many images in each categorys.

![image](https://github.com/baker-Xie/CarND-Traffic-Sign-Classifier/raw/master/sources/2.png)

### Design and Test a Model Architecture

As a first step, I didn't convert the RGB to grayscale because I think the RGB information is very useful.

Then, I decided to generate more training data because：
* I only get a 89% classification rate in the original image data and the original Le-Net.
* So I modified the Le-Net (increase the depth of filters and the size of Fully Connected Layer). Generally, a more complex network will get a higher accuracy.
* In order to prevent over-fitting and make the network more robust, I need to create more training examples.

To add more data to the the data set, I used translation because it's very easy to realize.

As a last step, I normalized the pixel value to [-1,1] so that the data has mean zero and equal variance. Generally everyone will do this step. But in my opinion I think this step has little effect on the final result because all pixels have the same value range: [0,255].

Here is an example of an original image and an augmented image:

![alt text][imgs/image3.jpg]

The difference between the original data set and the augmented data set is that the augmented data set have more images than the original data set.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x18  				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x48  				|
| Fully connected		| inputs 1200, outputs 360        									|
| RELU					|		
| Fully connected		| inputs 360, outputs 252        									|
| RELU					|		
| Fully connected		| inputs 252, outputs 43        									|
| Softmax				|  

To train the model, I used the following setting:
* Optimizer : Adam
* Batch Size : 128
* Learning Rate : 0.001
* Epochs : 200
* Drop out: No
* Regularization: No

When traning model, I find many problems, and my solutions are as follows:

* From the initial LeNet model taken from the Udacity course which was designed for MNIST problem I find it work well in the classification problem. So at this problem, I firstly chosed this model.
* When using the original Le-Net in 20 epochs, I can only get a 0.89 validation accuray. So I increase the epochs to 100, it's works. And then I increase the epochs to 200, it's performs better (0.93). When I set epochs to a number which is bigger than 200, the performance didn't increase any more, that means after 200 epochs, the model was convergent.
* The learning rate is very important.(0.001 performs better than 0.003)
* In order to get a better performance , I increased the model size because I think the model is under-fitting.
* Meanwhile, I get more fake data so that the model wont' get over-fitting. (0.953)

After the above steps, my final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.3% 
* test set accuracy of 93.8%

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)  									| 
| Bumpy road     			|Bumpy road										|
| Caution					| Caution											|
| Keep right	      		| Keep right			 				|
| No entry			| No entry     							|

The model correctly predicts all the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%.

For all images, the model are relatively sure that what they should be. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Speed limit (30km/h)   									| 
| .999     				| Bumpy road 										|
| .999					| Caution											|
| .999	      			| Keep right					 				|
| .999				    | No entry      							|
