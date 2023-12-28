import random
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import math
import sys

sample_size = 256 #enter length of number of instances per event (sample size = num_points in PointNet)
data_raw_where = '/home/DAVIDSON/dmkurdydyk/FRIB_Distant_Transfer/data/' # choose where to store raw data(before sampling)
data_sampled_where = '/home/DAVIDSON/dmkurdydyk/FRIB_Distant_Transfer/data/data_sampled/' # choose where to store sampled data
file = h5py.File('/home/DAVIDSON/dmkurdydyk/FRIB_Distant_Transfer/data/run210.h5', 'r') # select the location where your h5 data is stored

# The following will be needed to run later cells to align event ids
original_keys = list(file.keys()) # the .keys() function lists the labels in a dictionary
original_length = len(original_keys)

#making an array of the lengths of events
event_lens = np.zeros(original_length, int)

#For experimental data, some events in the h5 file might be empty, the following code remove empty events:

#count non-empty events
count = 0
#record non-empty events index
index = np.zeros(original_length, int)

for i in range(original_length):
    event = original_keys[i]
    event_lens[i-count] = len(file[event])
    if event_lens[i-count] < 5:
        count += 1
    else:
        index[i-count] = i
#remove empty event index
original_length = original_length - count


event_lens = event_lens[:original_length]
index = index[:original_length]

np.save(data_raw_where + 'event_lens_old.npy', event_lens)

#making an array of the events data-- [event #, instance, data value]
#length of each event is based on the longest event in dataset
#5th index now corresponds to index of event id in original_keys
# each instance will index according to the following 
# [0]x,[1]y,[2]z, [3]time, [4]Amplitude, [5]Event_id index

event_data = np.zeros((original_length, np.max(event_lens), 6), float) 

j = 0
for n in tqdm.tqdm(index):
    name = original_keys[n]
    event = file[name]
    ev_len = len(event)
    #converting event into an array
    for i,e in enumerate(event):
        instant = np.array(list(e))
        event_data[j][i][:3] = np.array(instant[:3]) #X, Y, Z
        event_data[j][i][3] = np.array(instant[4]) #Amplitude (charge)
        event_data[j][i][4] = ev_len # #points in events 
        event_data[j][i][-1] = int(n) #event Id
    j += 1
np.save(data_raw_where + 'more_than_5_old', event_data)

#Randomly choose sample_size of events
event_lens = np.load(data_raw_where + 'event_lens_old.npy')
data = np.load(data_raw_where + 'more_than_5_old.npy')
#insert desired array to sample from 

new_data = np.zeros((original_length, sample_size, 6), float) 
for i in tqdm.tqdm(range(original_length)):
    instant = 0
    ev_len = event_lens[i]    #length of event-- i.e. number of instances
    random_points = np.random.choice(range(ev_len), sample_size, replace = True if sample_size > ev_len else False)    #choosing the random instances to sample
    for r in random_points:
        new_data[i,instant] = data[i,r]
        instant += 1
np.save(data_sampled_where + str(sample_size)+'_sampled_old', new_data)      #creating new dataset within the h5 file for the event