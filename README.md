# === WEEK 1 ===

## M5 Project: Scene Understanding for Autonomous Vehicles
The goal of the project is to perform object detection, segmentation and recognition using Deep Learning in the context of scene understanding for autonomous vehicles. The network's goal will be to successfully compute the aforementioned tasks on classes like pedestrians, vehicles, road, etc.

<p align="center">
<img src="https://github.com/BourbonCreams/mcv-m5/blob/master/imgs/introduction.jpg" width="600"/>
</p>

## Team Info & Contacts

Team 9 (also known as למלם, pronounced "Lam Lam")

#### Team Members:

Lorenzo Betto: smemo23@gmail.com

Noa Mor: noamor87@gmail.com

Ana Caballero Cano: ana.caballero.cano@gmail.com

Ivan Caminal Colell: ivancaminal72@gmail.com

## Overleaf article
Link to the Overleaf [article](https://www.overleaf.com/read/rwdjpbdgsbdk), i.e. the report of the project.


#### Summaries of the two papers: <a href="https://drive.google.com/open?id=1M0HRZNI0OJJiaiefAOT1j8ABFqY55E2JLAgv--reY1E">VGG</a>, <a href="https://drive.google.com/open?id=1eKTcFKF5oGYx-GdWhsLea28V4AF49iJHj9ZHYdvrcas">SqueezeNet</a>.


# === WEEK 2 ===

## Task A: Run the code
#### Short abstract about what you implemented (5 lines max)
Task A: Bash file to output the number of samples of each folder.
Task B: Not implementable
Task Cii: We implemented a new CNN (LamLam) with two parallel sequential processes of convolutional layers.
Task E: We wrote the report

#### Short explanation of the code in the repository
Task A:
We have created a bash script that returns 3 txt (train, test, val) that contains a list
"subfolder_name; number_of_images".

Task B: Not implementable

Task Cii:

#### Results of the different experiments
Task A. Run the provided code
Analyze the dataset: 
The images are 64x64 pixels and differ in point of view, background, illumination and HUE. Furthermore, some images are slightly blurred.

Count the number of samples per class:
16527 for training, 
644 for validation 
8190 for testing. 

To know the number of samples per class follow the link:
<a href="https://drive.google.com/open?id=1NHeXsCl0G7QeRQZ1zyJq4GQdM6JjK0EQ1RyuQMIPJic">Google Sheets</a>

#### Accuracy of train/test
Accuracy Train: 97.7 %;  Accuracy Test: 95.2 %
The accuracy of train is better  than in the test set, as we expected.

#### For the this case which one provides better results, crop or resize?
On this dataset crop useless because images are already cropped, so resize it’s better.

####  Where does the mean subtraction takes place?
The mean subtraction takes place in the ImageDataGenerator, setting norm_featurewise_center to ‘True’.

#### Fine-tune the classification for the Belgium traffic signs dataset.
Custom accuracy:		Accuracy with Belgium traffic signs dataset:
Custom loss:			  Loss with Belgium traffic signs dataset:


## TASK B: Train a network on another dataset
Not implementable!

## Task Cii: Implement a new network

## Instructions for using the code

## Indicate the level of completeness of the goals of this week 

## Link to the Google Slide presentation
<a href="https://drive.google.com/open?id=1xdwzScs1yIeNa9y7kvcai-PCpIQG_0BUL5POIirqQiM">Slides Week 2 </a>

# Link to a Google Drive with the weights of the model 

