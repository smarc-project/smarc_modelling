# "Install" guide
For everything to run you are going to need to have Torch installed as well as the base SMaRC modelling libraries (numpy, matplotlib, pytest). If you do not have this then run:
```pip install torch```
Note that your hardware has to be CUDA capable if you are having problems with the install check the "Get Started" page for the pyTorch documentation.
All the code was developed using pyTorch 2.9.0 if you are facing any legacy issues consider downpatching to this version.

All save and load paths are "hardcoded" meaning that if you do not have the right file structure and you aren't running the code from the base directory it will results in an error.
To fix this either change the directories in the code itself or run it from the initial smarc_modelling folder.

You will also have to ensure that the ROS node "piml_msgs" is included properly in your workpage, this is crucial in order to load any data from the ROS bags.

# Models & Simulator

## Grid trainer
For running the grid trainer simply change what model you wish to train and its associated name in the SAM simulator and save name correspondingly.
PINN ==> "pinn" in SAM sim, "pinn.pt" in save name.
NN ==> "nn" in SAM sim, "nn.pt" in save name.
Naive NN ==> "naive_nn" in SAM sim, "naive_nn.pt" in save name.
From there you can change what parameters you wish to train over by adjusting the for loops and their corresponding values.

## Running the models
If you wish to run and evaluate the model run the file "piml_sim" this doubles both as the class instance for using the piml models in forwards prediction
but also where any plots themselves are made. You can also initialize a instance of this simulator by importing the class SIM and giving running that with the associated inputs.

Per standard all plotting is turned off in the piml_sim file so you would have to go in and enable whatever plot you wish to produce by changing its if statement to True.

# Utilities

## Post-Processing Guide

1. With data collected from the tank first check if the bag is using /mocap/sam_mocap2/ or the /mocap/sam_mocap/ topic and change
lines 46, 49 & 124 accordingly in add_timestamp.py by ensuring the topics there reflect what is in the bag.

2. Launch add_timestamp.py & sync_topics.py. In sync_topics you can adjust the "slop" which is the maximum allowed time difference between each data
point in a single synched message. ~0.2 seems to work very well for a good balance between quality and frequency of synched data.

3. Start a bag recording of the topic /synched_data this contains all the synched up datapoints in the bag with new timestamps
that are assigned as you play the bag you want to record. Note that this means that the original timestamp will be overwritten in case you would need that.

4. Play your bag, if everything is working correctly the command window you have the sync_topics running in should start printing "Published synched data!".

5. This new bag contains all the synched data with new timestamps and can then be loaded using load_data_from_bag or load_data_to_trajectory in utility functions, simply give it the name 
of the bag and wether you want the returned vectors to be in torch arrays of numpy vectors. Giving it "torch" will return torch any other input will return in numpy.
