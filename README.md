# Cotton-Disease-Detection
A data science project to detect weather cotton leaf or plant is a diseased one or fresh one by using transfer learning technique.

# Problem Statement

The problem is to detect wheather a given image of a Cotton Leaf or Cotton Plant is infected or is fine. if we can make a machine learning model then we can save a lot of human effort by detecting in advance which cotton leaves or plants are fine and which are not, we can do this by using drones to capture images automatically and then process it by machine learning model. Thus using technology in a createive way.

# Dataset

The dataset contains thousands of images of all 4 categories namely diseased cotton leaf, diseased cotton plant, fresh cotton leaf and fresh cotton plant, thus the model should have a good accuracy.

# Technology used

For this problem we will be using Transfer Learning with ImageDataGenerator. Generally VGG16 and VGG19 gives the best accuracy, but here we will be using Inception V3, as it is light and faster and giving better accuracy. This project is done in Tensorflow 2.3

# Webapp Demo

Please find the link to webapp for this project https://cotton-disease-detection.herokuapp.com/

# How to use Webapp

* Click on Upload image 
* Download images given in this github page, with names test1.jpg, test2.jpg etc 
* Upload these images to webapp 
* Check the result

# Conclusion

The model gives approximately 93% accuracy which is a very good accuracy for this kind of problem, thus the model is successful.
