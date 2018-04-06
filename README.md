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
<a href="https://drive.google.com/open?id=1prZl1Nyk6i8_jCgN52oghrHRBnsVwyZg"> Weights of the trained models </a>

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
Task A: We fixed some errors to be able to run the code. <br/>
Task B: We red  two articled and did a summary. <br/>
Task C: SSD object detector, using Keras <br/>
Task D: Evaluation of udacity dataset <br/>
Task E: Boost the performance throught data augmentation <br/>

## Short explanation of the code in the repository 
Task A: YOLO object detector <br/>
We fixed some errors to be able to run the code. We got <a href="https://drive.google.com/open?id=1V9YR25Qb4yf7Gs2LdtebVMVDHxpPV9RT3-6cwqTHtAA">this results </a>  <br/>

The dataset was also analyzed: <br/>
The number of signs in the annotation files do not always includes all the traffic sign exist in the image.  <br/>
They differ in the number of traffic signs, the orientation and illumination. <br/>

Task B: - <br/>
Task C: SSD object detector, Keras implementation<br/>
Task D: Run Udacity dataset for epochs to 40 and tune the colors (saturation) to solve the challenges of the dataset. <br/>
Task E: Data augmentation <br/>

## Results of the different experiments 
<a href="https://drive.google.com/open?id=1V9YR25Qb4yf7Gs2LdtebVMVDHxpPV9RT3-6cwqTHtAA"> All experiments results </a>  <br/>

## Instructions for using the code 
CUDA_VISIBLE_DEVICES=0 python train.py -c config/dataset.py -e expName

## Indicate the level of completeness of the goals of this week 
100%

## Link to a Google Drive with the weights of the trained models
<a href="https://drive.google.com/open?id=1cU9eV9GEl9Gmcz_LMSCrtbELU0tlOxcv">  Weights of the trained models </a>  <br/>

## Tasks:
### Task A Run the provided code
We fixed some errors to be able to run the initial code.<br/>
From the <a href="https://drive.google.com/open?id=1v05V3nCyqD8BxZZoJI0pJf1MvlsEqxU-eOVMx6MzBgA"> dataset analysis </a> we can conclude that: <br/>

TT100k dataset:  The number of signs in the annotation files do not always includes all the traffic sign that exist in the image, the selective choices are not clear. The images differ in the number of traffic signs, orientation and illumination.<br/>
Udacity dataset: consist of urban images while the camera always facing toward to road including the dashboard. There is a big difference between the train and test images. <br/>
Training images are in strong mid-day light, mostly saturated colors, shades and reflective light (e.g. reflective light from the windshield). While test images have more vivid colors in different time of day. <br/>


There is a large variance in the luminance in the photos, has a lot of un balance and disorders in the luminance (for example : reflective light from the windshield).
 <br/>
Propose (and implement) solutions. <br/>
A solution can be to pre process the images - tunning the colors (saturation). <br/>
Or training using data augmentation on the color channel - creating more variance in the color spectrum. <br/>

### Task B Read two papers
Summaries:  <br/>
<a href="https://drive.google.com/open?id=1Af6ICi9XkcP4RK567LYebVJcOn5t4dDbt4DXdzS0ey0">YOLO: You only Look Once </a>  <br/>
<a href="https://drive.google.com/open?id=1EUirquk_4uj3BQyfaq3AbGwjbZaH-LbeIcjtPwktpX8">SSD: Single Shot MultiBox Detector </a> <br/>

### Task Ci Implement a new network
Implementation found in this <a href="https://github.com/rykov8/ssd_keras">Github repo</a>.

### Task D Train the networks on a different dataset
Set-up new experiments files to detect among cars, pedestrians, and trucks on the Udacity dataset, Train and evaluate <br/>
Increment the number of epochs to 40.
We ran the yolo with plain udacity (for comparison with TT100K) <br/><br/>


### Task E Boost the performance of your network
We boosted the performance of the network by implementing the previous solution (pro process images changing color saturation) and applying data augmentation. <br/>

# === WEEK 5-6 ===
## Short abstract about what you implemented (5 lines max) 
Task (a): Run the provided code (YOLO-Traffic Sign)<br/>
Task (B): Read two papers: FCN for Semantic Segmentation and U-Net.<br/>
Task (c): Implement a new network, modifying a keras implemented<br/>
Task (d): Train the network on a different dataset <br/>

## Short explanation of the code in the repository 
Task (a): We used the preconfigured experiment file (camvid_segmentation.py) to segment objects with the FCN8 model.
Then we <a href= "https://drive.google.com/open?id=1DIopyfM3Wg2niLuBZgtusZNMgLWBS2WCZVhSy03qhpo"> analysed </a> the dataset and  evaluate  on train, val and test sets.<br/>
Task (B): -<br/>
Task (c): We implemented a new network,U-Net, modifying a keras implemented.<br/>
Task (d): Train the network on a different datasets, kitti, cambid, Cityscapes and Synthia_rand_cityscapes <br/>

## Results of the different experiments 
<a href="https://drive.google.com/open?id=14HRC3ptKjcdsh1dP8n4EuXx_JTnwh46K6ttrwsW0Xks"> All experiments results </a>  <br/>

## Instructions for using the code 
CUDA_VISIBLE_DEVICES=0 python train.py -c config/dataset.py -e expName <br/>

## Indicate the level of completeness of the goals of this week 
100%

## Link to a Google Drive with the weights of the model
<a href="https://docs.google.com/presentation/d/1CQr7opcnp8sZqT4n6qzXDiB4zvOCfZbNm3vmb70mtBw/edit?usp=sharing"> Google Slides </a>

## Tasks:
### Task A Run the provided code
We fixed some errors to be able to run the initial code
From the <a href="https://drive.google.com/open?id=1DIopyfM3Wg2niLuBZgtusZNMgLWBS2WCZVhSy03qhpo"> dataset analysis </a> we can conclide that: <br/>

Kitti and Camvid have the same classes. <br/>
Kitti dataset: the images are taken at the same time of day and there is an important contrast between sun and shadow, making it difficult to segment them. <br/>
Camvid dataset: the streets are more similar, but differ at the time of day, that we can observe in the lighting. <br/>

Cityscapes and Synthia_rand_cityscapes have the same classes. <br/>
Cityscapes dataset: it is similar in terms of illumination and color even though they are from different german cities. These images differ more in content (pedestrians, bicycles, vehicles, etc) <br/>
Synthia_rand_cityscapes differs on the scene. There are groups of the same scene which differ on the moment of the day. We can observe it on the illumination, shadows or even the rain.


### Task B Read two papers
<a href="https://drive.google.com/open?id=1VYtMOV2C7_Hu8RWosBKHoK6gU4sEQEzX9hx0OHHMS_I"> FCN </a>  and 
<a href="https://drive.google.com/open?id=1edZgYJHvHG2pP6HB3JyrsoBnbzMBM34BMMxj2tPnSik"> U-Net </a>

### Task C Implement a new network **
### Task D Train the networks on a different dataset **
### Task E Boost the performance of your network **
	
