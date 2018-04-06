# === WEEK 1 ===
## Title of the project 
M5 Project: Scene Understanding for Autonomous Vehicles

## Name of the group 
Team 9 (also known as למלם, pronounced "Lam Lam")

## Name and contact email of all the team members 
Lorenzo Betto: smemo23@gmail.com <br/>
Noa Mor: noamor87@gmail.com  <br/>
Ana Caballero Cano: ana.caballero.cano@gmail.com  <br/>
Ivan Caminal Colell: ivancaminal72@gmail.com  <br/>

## Short abstract about the project (2 lines max) 
The goal of the project is to perform object detection, segmentation and recognition using Deep Learning in the context of scene understanding for autonomous vehicles. The network's goal will be to successfully compute the aforementioned tasks on classes like pedestrians, vehicles, road, etc. <br/>

<p align="center">
<img src="https://github.com/BourbonCreams/mcv-m5/blob/master/imgs/introduction.jpg" width="600"/>
</p>

## Link to the overleaf 
Link to the Overleaf [article](https://www.overleaf.com/read/gdvggkpbcnmd), i.e. the report of the project.

## Summary of the two papers 
 <a href="https://drive.google.com/open?id=1M0HRZNI0OJJiaiefAOT1j8ABFqY55E2JLAgv--reY1E">VGG</a>, <a href="https://drive.google.com/open?id=1eKTcFKF5oGYx-GdWhsLea28V4AF49iJHj9ZHYdvrcas">SqueezeNet</a>.
 
## Tasks:
### Task A Understand the project
We read and understand the slides and Overleaf document 

### Task B Form teams
Team members are those named above.

### Task C Install and setup the development framework
We installed the necessary software.

# === WEEK 2 ===
## Short abstract about what you implemented (5 lines max) 
Task A: Bash file to output the number of samples of each folder. <br/>
Task B: We ran the code for KITTI dataset, for training and validation  <br/>
Task Cii: We implemented a new CNN (LamLam) with two parallel sequential processes of convolutional layers.  <br/>
Task E: We wrote the report

## Short explanation of the code in the repository 
Task A: We have created a bash script that returns 3 txt (train, test, val) that contain a list "subfolder_name; number_of_images".  <br/>
Task B: We ran the code for the KITTY, for trainning and validation. Not for test  <br/>
Task Cii: Our own CNN implementation, we named it LamLam (as our team).
It has two parallel sequential processes of convolutional layers of different sizes that allow to capture two different types of information.

## Results of the different experiments 
<a href="https://drive.google.com/open?id=1YQxsxH6de6iRGwwNZ_ifAmFJf_QR4kFX_DB-YkuoMaM"> All experiments results </a>
## Instructions for using the code 
CUDA_VISIBLE_DEVICES=0 python train.py -c config/dataset.py -e expName

## Indicate the level of completeness of the goals of this week
100%

## Link to the Google Slide presentation 
<a href="https://drive.google.com/open?id=1xdwzScs1yIeNa9y7kvcai-PCpIQG_0BUL5POIirqQiM">Slides Week 2 </a>

## Link to a Google Drive with the weights of the model
<a href="https://drive.google.com/open?id=1prZl1Nyk6i8_jCgN52oghrHRBnsVwyZg"> Weights </a>

## Tasks:
### Task A Run the provided code
- Analyze the dataset:
The images are 64x64 pixels and differ in point of view, background, illumination and HUE. Furthermore, some images are slightly blurred.

- Count the number of samples per class:
16527 for training,  <br/>
644 for validation   <br/>
8190 for testing.  <br/>

To know the number of samples per class follow the link:  <br/>
<a href="https://drive.google.com/open?id=1NHeXsCl0G7QeRQZ1zyJq4GQdM6JjK0EQ1RyuQMIPJic">Google Sheets</a>

- Accuracy of train/test
Accuracy Train: 97.7 %;  <br/>
Accuracy Test: 95.2 %  <br/>
The accuracy of train is better than in the test set, as expected.

- For this case which one provides better results, crop or resize?
On this dataset crop useless because images are already cropped, so resize is better.

-  Where does the mean subtraction takes place?
The mean subtraction takes place in the ImageDataGenerator, setting norm_featurewise_center to ‘True’.

- Fine-tune the classification for the Belgium traffic signs dataset.
Custom accuracy:		 <br/>
Accuracy with Belgium traffic signs dataset:  <br/>
Custom loss:			 <br/>
Loss with Belgium traffic signs dataset:  <br/>


### Task B Train the network on a different dataset
We ran the KITTI dataset for the training and the validation datasets since the test set is private and we can't access it.

### Task Cii Implement a new network
We used a CNN that was tested in the Machine Learning course of the same Master program. Such architecture is shown in
<p align="center">
<img src="https://github.com/BourbonCreams/mcv-m5/blob/master/imgs/CNN_LamLam.PNG" height=500"/>
</p>

and it performed well with a classification problem that involved scenery images. <br/>

The idea that led to the development of a network with two parallel sequential processes of convolutional layers of different sizes was to allow to capture two different types of information, the first one being the small details and texture and the second one to capture the composition and details in the bigger picture. <br/>

The model's parameters were optimized using a random search when the model was first used, i.e. in the Machine Learning course.

### Task D Boost the performance of your network
We boost the performance of the network by using a SPP layer (Spatial Pyramid Pooling) instead of a costum pooling layer in the end of each tower (to concatenate the two towers, their shape must agree).<br/>
In addition this layer makes the model independent from the image size.<br/>
The Training is done over TT100K dataset and testing is done over the Belgium database. On the way to try to create a generic model. <br/>
<p align="center">
<img src="https://github.com/BourbonCreams/mcv-m5/blob/master/imgs/CNN_Boost_LamLam.png" height=500"/>
</p>
												    
# === WEEK 3-4 ===
## Short abstract about what you implemented (5 lines max) 
## Short explanation of the code in the repository 
## Results of the different experiments 
## Instructions for using the code 
## Indicate the level of completeness of the goals of this week 
## Link to a Google Drive with the weights of the trained models
## Tasks:
### Task A Run the provided code
### Task B Read two papers
### Task C Implement a new network
### Task D Train the networks on a different dataset
### Task E Boost the performance of your network

# === WEEK 5-6 ===
## Short abstract about what you implemented (5 lines max) 
## Short explanation of the code in the repository 
## Results of the different experiments 
## Instructions for using the code 
## Indicate the level of completeness of the goals of this week 
## Link to a Google Drive with the weights of the model
## Tasks:
### Task A Run the provided code
### Task B Read two papers
### Task C Implement a new network
### Task D Train the networks on a different dataset
### Task E Boost the performance of your network
	
