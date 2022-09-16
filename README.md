# PointNet model for fission detection
Usage of latent space from PointNet model(trained on ModelNet10 data) to search for fission events  in data from FRIB
# How to use 
## Weights
There is a few steps:
Download and run the `Weights.py` file which will generate the weights you will use later on. 
You'll need to specify the following parameters:

- `weights_path` - This is where the weights generated by the PointNet model will be stored.
The path must be in the format `where_to_store_weights\`

- `num_points` -  the number of points on which the PointNet model will be trained, it must match
with `sample_size`. Default is `num_points = 512`, but actually there is no big difference(u can also try 256,128 or 64)

- `d_max`  - latent space size, default = `1024`

- `batch` - batch size, default and the best is `1`, don't change it if you don't want worse results

- `epochs` - number of epochs, default = `20`, when this number changes upwards, no changes occur, I advise you to touch it solely out of your curiosity

- `chair_t` - clarifies which specific two classes you will take for training, default = `dresser_chair`, you can try another two classes, but there is no point

### What the idea behind this stage:

We are training a PointNet model that was designed for a completely different task than ours - recognizing chairs, chests of drawers and tables. 
Our goal is to get the trained weights from this model, and use them in our model, with the same architecture, to create a hidden space
In the original article for this model, there were 10 classes, but for our purposes I reduced their number to two 
(you can choose which two specifically by changing the parameter `chair_r`)

## Data_Sampling

The purpose of this stage is to convert data from h5 format to numpy array, and also to remove junk data (data with less than 5 points) 
and sample the data.

u'll need to specify the following parameters:

- `sample_size` - must match with `num_points`, default = `512` 

- `data_raw_where` -  choose where to store raw data(before sampling). The path must be in the format `where_to_store_raw_data\`

- `data_sampled_where` - choose where to store sampled data. The path must be in the format `where_to_store_sampled_data\`

- `where_h5` - full path of where your h5 data is stored

After this stage you'll have sampled data, with the following dimensions:

`data_sampled` = `(num_of_events, sample_size, (X, Y, Z, Charge, num_points_in_event, event_id))`, where `event_id` - id of event in original h5 file

## Prediction 

The essence of the stage is as follows: to push the data we have into a two-class PointNet model, 
but not to look at the results it gives, but simply to get an intermediate layer - 1DMaxPulling (that is our `latent space`). 
Then we will put this intermediate layer in OneClassSVM, and we will get our predictions. That's it

You'll need to specify the following parameters:

- `data_raw_where` - where the raw data(before sampling) were stored

- `data_sampled_where` - where the sampled data were stored

- `weights_path` - where the weights were stored

- `predict_where` - Where to save predictions

- `predict_id_where` - Where to save id of fission event

- `umap_where` -  where to save umap plots(a special way to represent what the latent space looks like, it is a kind of convolution by coordinates)

- `sample_size` - default = `512`,( the one u use before in `Weights` and `Data_Sampling`

- `d_max` - default = `1024`  .Size of latent Space( the one u used before in `Weights`)

- `h5data` - default = `512_sampled`. What size of sampling do you use, for example: h5data = 512_sampled, this means that you use sample_size = 512, 
and this nead to match with sample_size that you defined earlier ( you don't need to write this things, just uncomment one line of code)

- `weights_type` - default = `dresser_chair_512_1024 `define what types of weights you are gonna use. It depends on what kind of wheights did you train. 
a little clarification: `512` here is `sample_size` and `1024` - size of latent space `d_max`

- `data_type` - default = `without_charge`. In case  you become interested in what will happen if you throw out part of the features from the dataset
(for example you can thrown away X axis, and use onle Y and Charge to make a predictions. But actually it is useless)

- `kernel` - default(and the best) = `rbf`. Defines the kernel parameter for `OneClassSVM`.

- `gamma` - default = `auto`. Another parameter for `OneClassSVM`. Actually there is no big difference between gamma = `auto` or gamma = `scale`

- `degree` - degree of kernel = `poly`. I did't use this much, because perfomance of `poly` not so good as `rbf`

- `nu` - default = `[0.01, 0.015, 0.02, 0.027, 0.03, 0.04, 0.06, 0.09]`, parameter nu for OneClassSVM, defines a lower bound on the number of points in one class. For example `nu = 0.01` means the at least `1%` of data needs to be in one class. nu is a nuppy array for the convenience of training and comparing results, since this way we can see how the model will behave with different nu parameters. And we need to remember that we have about 2.7 percent of all data are fission events(that correspond to `nu = 0.027`),  so the nu parameter should not be too large, otherwise we risk very badly worsening our results in Precision.

- `num_of_train_exmp` - default = `[1000,2000,10000]`, the number of events on which we train OneClassSVM. It turns out that it is not always expedient to train OneClassSVM at once on all events, sometimes it is more profitable to train OneClassSVM only on a part of events, for example, on 1000, 2000, or 10000 events. The advantage of this approach is that when training, for example, on 1000 events, we get more Recall than if we trained on 10,000 events, but at the same time, the Precision of a model that was trained on 10,000 events is more than that of a model that trained on 1000, so you need to clearly understand what goals you want to achieve. So I advise you to experiment with these parameters.

More information about OneClassSVM u can find in this article: https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72 or in this https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989

## Accuracy 

Code to check if the model is doing well. Since we are dealing with unlabeled data, we will carry out the check in a rather rough way - we will say that fission events are those events in which there are more than or equal to `100 points`. And we will compare our predictions and these labels to create confusing matrices and build dependency plots Recall(nu), Precision(nu), F1-score(nu)

u'll need to set next parameters:

- `data_raw_where` - where the raw data(before sampling) were stored

- `data_sampled_where` - where the sampled data were stored

- `weights_path` - where the weights were stored

- `predict_where` - where the predictions were stored

- `confusion_where` - where to save confusion matrices 

- `accuracy_where` - where to save accuracy ( plots Recall(nu), Precision(nu), F1-score(nu))

- `sample_size`  - sample size, the one u use before in `Prediction`

- `h5data` - what type of sampled data did u use, the same as in `Prediction`

- `eval_t` - default = `less`, type of data which you use for evaluation. There is 2 options, first is `less` that means that you defined not-fission 
events as events with less than `100` points, second is `angus` - this is labeled data (about 2500 events) from `Angus Wong`

- `data_type` - default = `dresser_chair_512_1024`, the same as before in `Prediction`

- `kernel`, `gamma`, `nu`, `degree`, `num_of_train_exmp` - the same as in `Prediction`








