# Utilities

## Post-Processing Guide

1. With data collected from the tank first check if the bag is using /mocap/sam_mocap2/ or the /mocap/sam_mocap/ topic and change
lines 46, 49 & 124 accordingly in add_timestamp.py by ensuring the topics there reflect what is in the bag.

2. Launch add_timestamp.py & sync_topics.py. In sync_topics you can adjust the "slop" which is the maximum allowed time difference between each data
point in a single synched message. ~0.2 seems to work very well for a good balance between quality and frequency of synched data.

3. Start a bag recording of the topic /synched_data this contains all the synched up datapoints in the bag with new timestamps
that are assigned as you play the bag you want to record. Note that this means that the original timestamp will be overwritten in case you would need that.

4. Play your bag, if everything is working correctly the command window you have the sync_topics running in should start printing "Published synched data!".

5. This new bag contains all the synched data with new timestamps and can then be loaded using load_data_from_bag in utility functions, simply give it the name 
of the bag and wether you want the returned vectors to be in torch arrays of numpy vectors. Giving it "torch" will return torch any other input will return in numpy.