# PointNet model for fission detection
Usage of latent space from PointNet model(trained on ModelNet10 data) to search for fission events  in data from FRIB
# How to use 
## Weights
There is a few steps:
Download and run the `Weights.py` file which will generate the weights you will use later on. 
You'll need to specify the following parameters:
`weights_path` - This is where the weights generated by the PointNet model will be stored.
The path must be in the format `where_to_store_weights\`

`num_points` -  the number of points on which the PointNet model will be trained, it must match
with `sample_size`. Default is `num_points = 512`, but actually there is no big difference(u can also try 256,128 or 64)

`d_max`  - latent space size, default = `1024`

`batch` - batch size, default and the best is `1`, don't change it if you don't want worse results

`epochs` - number of epochs, default = `20`, when this number changes upwards, no changes occur, I advise you to touch it solely out of your curiosity

`chair_t` - clarifies which specific two classes you will take for training, default = `dresser_chair`, you can try another two classes, but there is no point

### What the idea behind this stage:

We are training a PointNet model that was designed for a completely different task than ours - recognizing chairs, chests of drawers and tables. 
Our goal is to get the trained weights from this model, and use them in our model, with the same architecture, to create a hidden space
In the original article for this model, there were 10 classes, but for our purposes I reduced their number to two 
(you can choose which two specifically by changing the parameter `chair_r`)

## Data_Sampling

The purpose of this stage is to convert data from h5 format to numpy array, and also to remove junk data (data with less than 5 points) 
and sample the data.

u'll need to specify the following parameters:

`sample_size` - must match with `num_points`, default = `512` 

`data_raw_where` -  choose where to store raw data(before sampling). The path must be in the format `where_to_store_raw_data\`

`data_sampled_where` - choose where to store sampled data. The path must be in the format `where_to_store_sampled_data\`

`where_h5` - full path of where your h5 data is stored

After this stage you'll have sampled data, with the following dimensions:

`data_sampled` = `(num_of_events, sample_size, (X, Y, Z, Charge, num_points_in_event, event_id))`

## Prediction 

The essence of the stage is as follows: to push the data we have into a two-class PointNet model, 
but not to look at the results it gives, but simply to get an intermediate layer - OneDimMaxPulling (`latent space`). 
Then we will put this intermediate layer in OneClassSVM, and we will get our predictions. That's it

You'll need to specify the following parameters:

`data_raw_where` - where the raw data(before sampling) were stored

`data_sampled_where` - where the sampled data were stored

`weights_path` - where the weights were stored

`predict_where` - Where to save predictions

`umap_where` -  where to save umap plots(a special way to represent what the latent space looks like, it is a kind of convolution by coordinates)

`sample_size` - default = `512`,( the one u use before in `Weights` and `Data_Sampling`

`d_max` - default = `1024`  .Size of latent Space( the one u used before in `Weights`)

`h5data` - default = `512_sampled`. What size of sampling do you use, for example: h5data = 512_sampled, this means that you use sample_size = 512, 
and this nead to match with sample_size that you defined earlier ( you don't need to write this things, just uncomment one line of code)

`weights_type` - default = `dresser_chair_512_1024 `define what types of weights you are gonna use. It depends on what kind of wheights did you train. 
a little clarification: `512` here is `sample_size` and `1024` - size of latent space `d_max`

`data_type` - default = `without_charge`. In case  you become interested in what will happen if you throw out part of the features from the dataset
(for example you can thrown away X axis, and use onle Y and Charge to make a predictions. But actually it is useless)













