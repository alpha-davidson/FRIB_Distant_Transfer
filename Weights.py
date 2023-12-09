import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)
#### LOAD DATASET ####
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

############
#chair_t = 'chair_bathtub'
chair_t = 'dresser_chair' # 2 classes that we use to train PointNet model, there is no much difference in which two classes do you choose 
#chair_t = 'sofa_table'
#chair_t = 'chair_6'
d_max = 1024 #  size of latent space 
batch = 1
BATCH_SIZE  = batch # batch size, the best results for batch = 1
num_points = 256  # num_points = sample_size 
NUM_CLASSES = 2
NUM_POINTS = num_points
epochs = 20 # number of epochs

weights_path = 'weights/' # where to save weights

def parse_dataset(num_points=num_points):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    
    
    class_map = {}
    folders_1 = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
    #folders = [None]*2
    folders = [None]*2
    if chair_t == 'chair_bathtub':
        folders[0] = folders_1[2] #chair
        folders[1] = folders_1[4] #bathtub
    elif chair_t == 'dresser_chair':
        folders[0] = folders_1[0] #dresser
        folders[1] = folders_1[2] #chair
    elif chair_t == 'sofa_table':
        folders[0] = folders_1[3] #sofa
        folders[1] = folders_1[6] #table
    elif chair_t == 'chair_6':
        folders[0] = folders_1[0] 
        folders[1] = folders_1[1] 
        folders[2] = folders_1[2] 
        folders[3] = folders_1[3] 
        folders[4] = folders_1[4] 
        folders[5] = folders_1[5] 


    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points)) #Breaking the image into points 
            train_labels.append(i)                                  # and add it to array

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

#########################


NUM_POINTS = num_points
    



train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)


#########################

def augment(points, label):                                                   # Перетасовываем наши данные 
    # jitter points                                                           
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    
        
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)


x = conv_bn(x, 64)
x = conv_bn(x, 64)
x = tnet(x, 64)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
if d_max == 1024:
    x = conv_bn(x, 1024)
elif d_max == 512:
    x = conv_bn(x, 512)
elif d_max == 2048:
    x = conv_bn(x, 2048)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 512)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x) ### sigmoid -> softmax

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()


######### TRAIN MODEL ##############

# Output layer
if NUM_CLASSES == 2:
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Single output neuron for binary classification
    loss_function = "binary_crossentropy"
else:
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)  # Multiple output neurons for multi-class
    loss_function = "sparse_categorical_crossentropy"

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

# Compile model
model.compile(
    loss=loss_function,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

weights_chair = model.get_weights()
if NUM_CLASSES == 10:
    np.save(weights_path +  'weights_chair', weights_chair)
elif NUM_CLASSES == 2:
    if chair_t == 'chair_bathtub':
        np.save(weights_path + 'weights_chair_2', weights_chair)  
    elif chair_t == 'dresser_chair':
        if num_points == 2048:
            if batch == 32:
                 np.save(weights_path + chair_t + '_2048_32', weights_chair)
            elif batch ==16:
                 np.save(weights_path + chair_t + '_2048_16', weights_chair)
        elif num_points == 1024:
            if batch == 16:
                 np.save(weights_path + chair_t + '_1024_16', weights_chair)
            if batch == 1:
                 np.save(weights_path + chair_t + '_1024_1', weights_chair)
        elif num_points == 512:
            if m_type == 'chair_s':
                np.save(weights_path + chair_t + '_512', weights_chair)
            elif m_type == 'chair_b':
                if batch == 32:
                    np.save(weights_path + chair_t + '_512_32', weights_chair)
                elif batch == 16:
                    np.save(weights_path + chair_t + '_512_16', weights_chair)
                elif batch == 8:
                    np.save(weights_path + chair_t + '_512_8', weights_chair)
                elif batch == 1:
                    if epochs == 20:
                        if d_max == 512:
                            np.save(weights_path + chair_t + '20_512_512', weights_chair)
                        elif d_max == 2048:
                            np.save(weights_path + chair_t + '20_512_2048', weights_chair)
                        elif d_max == 1024:
                            np.save(weights_path + chair_t + '20_512_1024', weights_chair)
                        else:
                            np.save(weights_path + chair_t + '20_512_1', weights_chair)
                    elif epochs == 30:
                        np.save(weights_path + chair_t + '_512_1_30', weights_chair)
                    
           
        elif num_points == 256:
            np.save(weights_path + chair_t + '_' + str(num_points)+ '_' + str(d_max), weights_chair)
        elif num_points == 375:
            np.save(weights_path + chair_t + '_375', weights_chair)
        elif num_points == 128:
            np.save(weights_path + chair_t + '_128_1024', weights_chair)
        elif num_points == 64:
            np.save(weights_path + chair_t + '_64_1024', weights_chair)
        elif num_points == 32:
            np.save(weights_path + chair_t + '_32_1024', weights_chair)
        
    elif chair_t == 'sofa_table':
        if batch == 1: 
             np.save(weights_path + chair_t + '_512_1', weights_chair)
elif NUM_CLASSES == 6:
    if chair_t == 'chair_6':
        np.save(weights_path + chair_t + '_512_1', weights_chair)