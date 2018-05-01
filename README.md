# 2018 Data Science Competition

## Overview
During the spring semester of my senior year at the University of Idaho, I took a Mathematics of Deep Learning course. As part of the course, the professors hosted a data science competition, open to all students (undergraduate and graduate) at the university. The competition had two parts: one problem dealing with machine learning and another contest for data visualization techniques. I participated in the machine learning contest.

## Problem
For the machine learning competition, we were tasked with designing a neural network to recognize handwritten digits and mathematical operators (+, -, =) with the end goal of grading simple equations comprised of those digits and operators.

### Training Data*
The training data set is an array with 80,000 rows and 577 columns. The first entry of each row is the index (or ID) of that row, and the remaining 576 entries are real numbers between 0.000 and 1.000, which represent signal intensity at the corresponding pixels. Once reshaped as a 24x24 array (that is, putting the first 24 entries in the first row, and the next 24 entries in the second row, and so on), these 576 entries represent a greyscale image of either a handwritten digit from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or a handwritten symbol from +, -, = with label a label from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 respectively. The true label of this image is recorded in the corresponding row of the training labels. 

The handwritten digits and symbols in this training set are different from but similar to the ones cropped from the expressions in the test set to be described below.

### Test Data*
The test data set is an array with 20,000 rows and 2881 columns. The first entry is the index (ID) of that row, and the remaining 2880 entries are real numbers between 0.000 and 1.000, representing the signal intensity at the corresponding pixel. Once reshaped as a 24x120 array, these 2880 entries represent a greyscale image of a handwritten arithmetic expression. These expressions are in one of the following forms:
- a+b=c
- a-b=c
- a=b+c
- a=b-c

where a, b, c are integers from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Some of the expressions are mathematically correct, and some are incorrect. The correct expressions have a true label 1 (integer), and the incorrect expression have a true label 0 (integer). 

## Implementations
It was recommended that we use TensorFlow and Convolutional Neural Networks (CNN) for this competition. I had not yet used either, but it seemed like a good opportunity to learn and familiarize myself with both.

Both models rely on the Adam optimizer and cross entropy loss calculation.

### Tensorflow 
- Branch: [master](https://github.com/trevmo/2018_dscomp)

Initially, I implemented a CNN in TensorFlow. The CNN had the following structure:
- Input layer
- Convolutional layer
- Max pooling layer
- Convolutional layer (twice as many filters as the first convolutional layer)
- Max pooling layer
- Dense layer
- Dropout layer
- Output layer (13 classes)

With this structure, and some trial and error with the hyperparameters, I was able to achieve 98.08% accuracy on the test data. 

### TFLearn 
- Branch: [tflearn](https://github.com/trevmo/2018_dscomp/tree/tflearn)

This branch is still in a Work-In-Progress state. After implementing my model in TensorFlow, I started researching ways to easily add data preprocessing and augmentation. I discovered that the TFLearn package and some built-in methods for such work as well as a similar structure and setup as TensorFlow. (Makes sense given that it's basically a wrapper of TensorFlow.)

The CNN had the following structure:
- Data Augmentation (rotations of up to 15 degrees)
- Input layer
- Convolutional layer
- Max pooling layer
- Batch normalization layer
- Convolutional layer (twice as many filters as the first convolutional layer)
- Max pooling layer
- Batch normalization layer
- Dense layer
- Dropout layer
- Output layer (13 classes)

While I did not finish the work on this branch within the scope of the competition, I was able to complete the model and experiment by training it with different combinations of data augmentation, batch normalization, dropout, etc. Since I did not fully incorporate its output into my Grader class, I instead analyzed how many epochs it took to reach different levels of validation accuracy on the training data.

It was an interesting exercise, exposing me to a new package and concepts. It would be interesting to complete the work at some point and see how it performs compared to the complete TensorFlow model/Grader combo. However, that would require access to the full test data set with labels.

### Grader class
Both models rely on a Grader class to perform the actual analysis required by the competition problem statement. This class processes the test data, dividing each equation image array into five separate images (corresponding to the numeric digits and mathematical operators comprising the equation). It then runs each image through the trained network, predicting the value or operator. From there, it checks the accuracy of the complete equation and logs the predicted answer to a file.

Initially, I structured the class so that it passed each individiual image to the network for prediction. However, this process was painfully slow -- taking 5+ hours to complete the test set of 20,000 equations. In order to speed up the process, I restructured it so that it would reshape the test data into a large matrix containing 100,000 images (1 image per row in the matrix / 5 images per equation). It would then predict the values for the entire test set in one pass. After the computing the predictions, the Grader class would loop over the results to grade each equation. This reduced the run time to ~35 minutes, with the bulk of that time spent on processing and reshaping the data. With a little more work, I am sure it could be further optimized.

## References
\* Taken from the competition [website](https://dscomp.ibest.uidaho.edu/data).

Thanks to [Dr. Frank Gao](http://www.webpages.uidaho.edu/~fuchang/) and [Dr. Linh Nguyen](http://webpages.uidaho.edu/lnguyen/) for teaching this class and organizing the competition!
