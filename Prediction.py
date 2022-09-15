import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import click 
import umap

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
    
    def get_config(self):
        return {'l2reg': self.l2reg,
               'num_features': self.num_features,
               'eye': self.eye.numpy().tolist()}
    
def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_bn(inputs, 64) #64->32
    x = conv_bn(x, 128) #128 -> 64
    x = conv_bn(x, 1024) #1024 -> 512
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512) # 512 -> 256
    x = dense_bn(x, 256) #256 -> 128
        
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def pnet(sem_seg_flag, num_points, num_classes, dimension):
    #Since current number of classes is about 1/4 of pointnet, used 1/4 of size for all layers except initial 3 for input (x,y,z)    
    inputs = keras.Input(shape=(num_points, dimension))
    a = tnet(inputs, dimension)
    #print(a.shape)
    
    b = conv_bn(a, 64) # 64 -> 32
    c = conv_bn(b, 64) # 64 -> 32
    d = tnet(c, 64) #64 -> 32
    e = conv_bn(d, 64) # 64 -> 32
    f = conv_bn(e, 128) # 128 -> 64
    if d_max == 1024:
        x = conv_bn(f, 1024) #1024 -> 512
    elif d_max == 512:
        x = conv_bn(f,512)
    elif d_max == 2048:
        x = conv_bn(f,2048)
    
    global_features = layers.GlobalMaxPooling1D(name='global_features')(x)
    #return global_features

    if sem_seg_flag:
        x = tf.expand_dims(global_features, axis=1) #x -> global_features
        #print(x.shape)
        x = tf.repeat(x, repeats=num_points, axis=1)
        long = layers.Concatenate(axis=2)([d,x])
        x = conv_bn(long, 512) 
        x = conv_bn(x, 256)
        x = conv_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        x = conv_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        #outputs = conv_bn(x, num_classes)
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="SemSegPointNet")
        model.summary()
    else:
        x = dense_bn(global_features, 512) # 512 -> 256
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 256) # 256 -> 128

        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="EventClassPointNet")
        model.summary()
    
    return model, global_features


data_raw_where = 'data/' # where the raw data(before sampling) were stored
data_sampled_where = 'data/data_' # where the sampled data were stored
weights_path = 'weights/' # where the weights were stored
predict_where = 'one_class_SVM_predict/' # Where to save predictions
umap_where = 'umap_plots/' # where to save umap plots

ISOTOPE = 'fission'
folder = 'datalessthan5/'


sample_size = 256 # sample size = num_points 
d_max = 1024 # Size of latent Space


# X, Y, Z, Charge, #points, Event Id
# first number - sample size (for ex. 512_sampled -> sample size = 512)
#h5data = '512_sampled' # X, Y, charge as features
#h5data = '512_sampled_x' # Y, Z, Charge as features
#h5data = '512_sampled_y' # X, Z, Charge as features 
#h5data = '512_sampled_n' # X, Y,  num_points as features
#h5data = '512_sampled_ncx' #X, Charge, num_points
#h5data = '512_sampled_ncz' #Z, Charge, num_points
#h5data = '64_sampled' # X, Y, Charge 
#h5data = '32_sampled' # X, Y, Charge 
#h5data = '32_sampled_x' # Y, Z, Charge
#h5data = 'without_tresh' # X, Y, Charge
#h5data = 'data_20'
#h5data = 'data_250'
#h5data = '128_sampled' # X, Y, Charge 
h5data = '256_sampled' #X, Y, Charge 
#h5data = 'with_n' # with  number of points (insted of x, for example)
#Checkpoint
if h5data == 'without_tresh':
    data_raw1 = np.load(folder + ISOTOPE + '_size' + str(sample_size) + '_sampled.npy')
    data_without_charge = np.delete(data_raw1, (2,3,5))
    
    data_without_charge_y = np.copy(data_without_charge)
    data_without_charge_y = data_without_charge_y[:,:,(0,2)]

    data_without_charge_x = np.copy(data_without_charge)
    data_without_charge_x = data_without_charge_x[:,:,(1,2)]

    data_without_charge_z = np.copy(data_without_charge)
    data_without_charge_z = data_without_charge_z[:,:,(0,1)]
elif h5data == 'with_tresh':
    data_raw1 = np.load(data_sampled_where + 'with_tresh_sampled.npy')
    data_without_charge = np.delete(data_raw1, (3,4,5,6), axis = 2)# with event id
elif h5data == 'data_sampled':
    data_raw1 = np.load(data_sampled_where + 'index_sampled.npy')
    data_without_charge = np.delete(data_raw1, (3,4,5,6), axis = 2)# with event i
elif h5data == '512_sampled':
    data_raw1 = np.load(data_sampled_where + '512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# X, Y, Charge
elif h5data == '512_sampled_x':
    data_raw1 = np.load(data_sampled_where+'512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (0,4,5), axis = 2)#  Y, Z, Charge
elif h5data == '512_sampled_y':
    data_raw1 = np.load(data_sampled_where+'512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (1,4,5), axis = 2)#  X, Z, Charge
elif h5data == '512_sampled_n':
    data_raw1 = np.load(data_sampled_where+'512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,3,5), axis = 2)#  X, Y, num_points
elif h5data == '512_sampled_ncx':
    data_raw1 = np.load(data_sampled_where+'512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (1,2,5), axis = 2)#  X, charge, num_points
elif h5data == '512_sampled_ncz':
    data_raw1 = np.load(data_sampled_where+'512_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (0,1,5), axis = 2)
elif h5data == '64_sampled':
    data_raw1 = np.load(data_sampled_where+'64_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# X, Y, Charge
elif h5data == '32_sampled':
    data_raw1 = np.load(data_sampled_where+'32_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# with event i
elif h5data == '32_sampled_x':
    data_raw1 = np.load(data_sampled_where+'32_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (0,4,5), axis = 2)# with event i
elif h5data == 'data_250':
    data_raw1 = np.load(data_sampled_where+'250_sampled.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# with event i
elif h5data == '128_sampled':
    data_raw1 = np.load(data_sampled_where+'128_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# X, Y, Charge
elif h5data == '256_sampled':
    data_raw1 = np.load(data_sampled_where+'256_sampled_old.npy')
    data_without_charge = np.delete(data_raw1, (2,4,5), axis = 2)# with event i


#weights_type = 'random'
#weights_type = 'Mg22_without_charge_y'
#weights_type = 'Mg22_without_charge_x'
#weights_type = 'Mg22_without_charge_z'
#weights_type = 'Mg22_without_charge_z_10'
#weights_type = 'Mg22'
#weights_type = 'chair_without_charge_x'
#weights_type = 'chair_without_charge_y'
#weights_type = 'chair_without_charge_z'
#weights_type = 'chair_2_without_charge'
#weights_type = 'chair_2_without_charge_x'
#weights_type = 'chair_2_without_charge_y'
#weights_type = 'chair_2_without_charge_z'
#weights_type = 'dresser_chair'
#weights_type = 'dresser_chair_without_charge_x'
#weights_type = 'dresser_chair_without_charge_1024'
#weights_type = 'dresser_chair_without_charge_512_16'
#weights_type = 'dresser_chair_without_charge_512_8'
#weights_type = 'dresser_chair_without_charge_512_1'
#weights_type = 'dresser_chair_250'
#weights_type = 'dresser_chair_512_512'
#weights_type = 'dresser_chair_512_2048'
#weights_type = 'dresser_chair_512_1024'
#weights_type = 'dresser_chair_512_1024_x'
#weights_type = 'dresser_chair_512_1024_y'
#weights_type = 'dresser_chair_512_1024_n' # 
#weights_type = 'dresser_chair_512_1024_ncx' # #points, charge and x
#weights_type = 'dresser_chair_512_1024_ncz'
#weights_type = 'dresser_chair_512_1024_old' # use old data + new weights to be sure that weights isok
#weights_type = 'dresser_chair_32_1024'
#weights_type = 'dresser_chair_128_1024'
weights_type = 'dresser_chair_256_1024'
#weights_type = 'dresser_chair_64_1024' # 64 - sample, 1024 - size of 1maxpool
#weights_type = 'dresser_chair_512_2048' #sample size + max p size
#weights_type = 'dresser_chair_375_1'
#weights_type = 'dresser_chair_512_1_512'
#weights_type = 'dresser_chair_512_1_2048'
#weights_type = 'dresser_chair_without_charge_x_512_1'
#weights_type = 'dresser_chair_without_charge_z_512_1'
#weights_type = 'dresser_chair_2_without_charge_512_1'
#weights_type = 'dresser_chair_without_charge_1024_16'
#weights_type = 'dresser_chair_without_charge_1024_1'
#weights_type = 'dresser_chair_without_charge_512'
#weights_type = 'dresser_chair_without_charge_256'
#weights_type = 'dresser_chair_without_charge_512_big'
#weights_type = 'dresser_chair_without_charge_512_big_2'
#weights_type = 'dresser_chair_without_charge_2048_32'
#weights_type = 'sofa_table'
#weights_type = 'sofa_table_without_charge_512_1'
#weights_type = 'chair_6_512_1'
#weights_type = 'dresser_chair_512_1_30'
#weights_type = 'Mg22_without_charge'
#data_type = 'with_charge'
data_type = 'without_charge'
#data_type = 'without_charge_y'
#data_type = 'without_charge_x'
#data_type = 'without_charge_z'

if data_type == 'with_charge':
    dimension = 4
elif data_type == 'without_charge':
    dimension = 3
elif data_type == 'without_charge_y':
    dimension = 2
elif data_type == 'without_charge_x':
    dimension = 2 
elif data_type == 'without_charge_z':
    dimension = 2 

    
    
num_points = sample_size

num_classes = 2 # 6 -> 2
nrand = 3

pretrain_model, global_features = pnet(sem_seg_flag = False, num_points = num_points, num_classes = num_classes, dimension = dimension)


#w_t = np.load('weights/weights_random_'+str(nrand)+'.npy', allow_pickle = True)
#pretrain_model.set_weights(w_t[:])

#model = keras.Model(inputs=pretrain_model.input, outputs=pretrain_model.get_layer('global_features').output, name="PointNet")

if weights_type == 'random':
    if data_type == 'without_charge':
        w_t = np.load(weights_path+'Random_weights_without_charge_'+str(nrand)+'.npy', allow_pickle = True)
elif weights_type == 'Mg22':
    w_t = np.load(weights_path+'weights_Mg22_np.npy', allow_pickle = True)
elif weights_type == 'Mg22_without_charge':
    w_t = np.load(weights_path+'22Mg_weights_without_charge.npy', allow_pickle = True)
elif weights_type == 'Mg22_without_charge_y':
    w_t = np.load(weights_path+'Mg22_weights_without_charge_y.npy', allow_pickle = True)
elif weights_type == 'Mg22_without_charge_x':
    w_t = np.load(weights_path+'Mg22_weights_without_charge_y.npy', allow_pickle = True) #the same as for without charge and Y 
elif weights_type == 'Mg22_without_charge_z':
    w_t = np.load(weights_path+'Mg22_weights_without_charge_y.npy', allow_pickle = True) #the same as for without charge and Y 
elif weights_type == 'chair':
    w_t = np.load(weights_path+'weights_chair.npy', allow_pickle = True)
elif weights_type == 'chair_without_charge_x':
    w_t = np.load(weights_path+'chair_weights_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'chair_without_charge_y':
    w_t = np.load(weights_path+'chair_weights_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'chair_without_charge_z':
    w_t = np.load(weights_path+'chair_weights_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'chair_2_without_charge':
    w_t = np.load(weights_path+'weights_chair_2.npy', allow_pickle = True)
elif weights_type == 'chair_2_without_charge_x':
    w_t = np.load(weights_path+'chair_weights_2_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'chair_2_without_charge_y':
    w_t = np.load(weights_path+'chair_weights_2_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'chair_2_without_charge_z':
    w_t = np.load(weights_path+'chair_weights_2_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'Mg22_without_charge_z_10':
    w_t = np.load(weights_path+'Mg22_weights_without_charge_y_10.npy', allow_pickle = True)
elif weights_type == 'dresser_chair':
    w_t = np.load(weights_path+'weights_dresser_chair.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_x':
    w_t = np.load(weights_path+'dresser_chair_without_charge_x.npy', allow_pickle = True)
elif weights_type == 'sofa_table':
    w_t = np.load(weights_path+'weights_sofa_table.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_1024':
    w_t = np.load(weights_path+'weights_dresser_chair_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512':
    w_t = np.load(weights_path+'weights_dresser_chair_512.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_256':
    w_t = np.load(weights_path+'weights_dresser_chair_256.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512_big':
    w_t = np.load(weights_path+'weights_dresser_chair_512_big.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512_big_2':
    w_t = np.load(weights_path+'weights_dresser_chair_512_big_2.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512_16':
    w_t = np.load(weights_path+'weights_dresser_chair_512_16.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_2048_32':
    w_t = np.load(weights_path+'weights_dresser_chair_2048_32.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512_8':
    w_t = np.load(weights_path+'weights_dresser_chair_512_8.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_512_1':
    if h5data == 'without_charge':
        w_t = np.load(weights_path+'weights_dresser_chair_512_1.npy', allow_pickle = True) 
    elif h5data == '512_sampled':
        w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True) 
elif weights_type == 'dresser_chair_2_without_charge_512_1':
    w_t = np.load(weights_path+'weights_dresser_chair_2_512_1.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_1024_16':
    w_t = np.load(weights_path+'weights_dresser_chair_1024_16.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_1024_1':
    w_t = np.load(weights_path+'weights_dresser_chair_1024_1.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_x_512_1':
    w_t = np.load(weights_path+'dresser_chair_without_charge_x_512.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_without_charge_z_512_1':
    w_t = np.load(weights_path+'dresser_chair_without_charge_x_512.npy', allow_pickle = True)
elif weights_type == 'sofa_table_without_charge_512_1':
    w_t = np.load(weights_path+'weights_sofa_table_512_1.npy', allow_pickle = True)
elif weights_type == 'chair_6_512_1':
    w_t = np.load(weights_path+'chair_6_512_1.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1_30':
    w_t = np.load(weights_path+'dresser_chair_512_1_30.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_375_1':
    w_t = np.load(weights_path+'dresser_chair_375.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1_512':
    w_t = np.load(weights_path+'dresser_chair_512_1_max_512.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1_2048':
    w_t = np.load(weights_path+'dresser_chair_512_1_max_2048.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_250':
    w_t = np.load(weights_path+'weights_dresser_chair_250.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_512':
    w_t = np.load(weights_path+'dresser_chair_512_512.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_2048':
    w_t = np.load(weights_path+'dresser_chair_512_2048.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_64_1024':
    w_t = np.load(weights_path+'weights_dresser_chair_64.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_32_1024':
    w_t = np.load(weights_path+'dresser_chair_32_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_32_1024_x':
    w_t = np.load(weights_path+'dresser_chair_32_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_128_1024':
    w_t = np.load(weights_path+'dresser_chair_128_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_256_1024':
    w_t = np.load(weights_path+'dresser_chair_256_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_old':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_x':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_y':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_n':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_ncx':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
elif weights_type == 'dresser_chair_512_1024_ncz':
    w_t = np.load(weights_path+'dresser_chair_512_1024.npy', allow_pickle = True)
pretrain_model.set_weights(w_t[:])

model = keras.Model(inputs=pretrain_model.input, outputs=pretrain_model.get_layer('global_features').output, name="PointNet")

if data_type == 'without_charge':
    train_feature = model.predict(data_without_charge, verbose=1)
    train_feature = StandardScaler().fit_transform(train_feature)
    
elif data_type == 'without_charge_y':
    train_feature = model.predict(data_without_charge_y, verbose=1)
    train_feature = StandardScaler().fit_transform(train_feature)
elif data_type == 'without_charge_x':
    train_feature = model.predict(data_without_charge_x, verbose=1)
    train_feature = StandardScaler().fit_transform(train_feature)
elif data_type == 'without_charge_z':
    train_feature = model.predict(data_without_charge_z, verbose=1)
    train_feature = StandardScaler().fit_transform(train_feature)

    

np.save('train_feat_output/train_feature_' + weights_type + 
        '_weights_', train_feature)

reducer = umap.UMAP()
embedding = reducer.fit_transform(train_feature)

train_feature_without_shuffle = np.copy(train_feature)

np.random.shuffle(train_feature)
#train_feature_shuffle = np.load('train_feat_output/Shuffle_train_feature.npy')

# Parameters for OneClassSVM, best results for kernel = rbf, gamma = auto
kernel = 'rbf' 
#kernel = 'linear'
#kernel = 'poly'
gamma = 'auto'
#gamma = 'scale'
#gamma = 0.00001
degree = 3

num_of_predict = data_without_charge.shape[0]

train_feature_without_shuffle = train_feature_without_shuffle[:num_of_predict]
train_feature = train_feature[:num_of_predict] # slice num_of_predict = 77659

nu = np.array([0.01, 0.015, 0.02, 0.027, 0.03, 0.04, 0.06, 0.09])

num_of_train_exmp = np.array([1000,2000,10000])
#num_of_train_exmp = np.array([data_without_charge.shape[0]])

for j in num_of_train_exmp:
    num_of_train_ex = j
    train_feature_without_shuffle = train_feature_without_shuffle[:num_of_predict]
    train_feature = train_feature[:num_of_predict]
    
    for i in nu:
        clf = OneClassSVM(gamma = gamma,kernel = kernel, 
                      nu = i, degree = degree ).fit(train_feature[:num_of_train_ex])

        predict_2 = clf.predict(train_feature_without_shuffle)
        
        if data_type == 'with_charge' and weights_type == 'Mg22':
            np.save(predict_where + 'Mg22_weights' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'with_charge' and weights_type == 'random':
            np.save(predict_where + 'Random_weights_nrand' + str(nrand) +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'Mg22_without_charge':
            np.save(predict_where + 'Mg22_weights_without_charge' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'random':
            np.save(predict_where + 'Random_weights_without_charge_nrand' + str(nrand) +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_y' and weights_type == 'Mg22_without_charge_y':
            np.save(predict_where + 'Mg22_weights_without_charge_y' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_x' and weights_type == 'Mg22_without_charge_x':
            np.save(predict_where + 'Mg22_weights_without_charge_x' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_z' and weights_type == 'Mg22_without_charge_z':
            np.save(predict_where + 'Mg22_weights_without_charge_z' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'chair':
            np.save(predict_where + 'Chair_weights_without_charge' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_x' and weights_type == 'chair_without_charge_x':
            np.save(predict_where + 'Chair_weights_without_charge_x' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_y' and weights_type == 'chair_without_charge_y':
            np.save(predict_where + 'Chair_weights_without_charge_y' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_z' and weights_type == 'chair_without_charge_z':
            np.save(predict_where + 'Chair_weights_without_charge_z' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'chair_2_without_charge':
            np.save(predict_where + 'Chair_weights_without_charge_2' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_x' and weights_type == 'chair_2_without_charge_x':
            np.save(predict_where + 'Chair_weights_without_charge_x_2' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_z' and weights_type == 'Mg22_without_charge_z_10':
            np.save(predict_where + 'Mg22_weights_without_charge_z' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_y' and weights_type == 'chair_2_without_charge_y':
            np.save(predict_where + 'Chair_weights_without_charge_y_2' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_z' and weights_type == 'chair_2_without_charge_z':
            np.save(predict_where + 'Chair_weights_without_charge_z_2' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair':
            np.save(predict_where + 'dresser_chair_weights_without_charge' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'sofa_table':
            np.save(predict_where + 'sofa_table_weights_without_charge' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_x' and weights_type == 'dresser_chair_without_charge_x':
            np.save(predict_where + 'dresser_chair_weights_without_charge_x' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024':
            np.save(predict_where + 'dresser_chair_weights_without_charge_1024' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512':
            np.save(predict_where + 'dresser_chair_weights_without_charge_512' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_256':
            np.save(predict_where + 'dresser_chair_weights_without_charge_256' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_big':
            np.save(predict_where + 'dresser_chair_weights_without_charge_512_big' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_big_2':
            np.save(predict_where + 'dresser_chair_weights_without_charge_512_big_2' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_16':
            np.save(predict_where + 'dresser_chair_weights_without_charge_512_16' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_2048_32':
            np.save(predict_where + 'dresser_chair_weights_without_charge_2048_32' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_8':
            np.save(predict_where + 'dresser_chair_weights_without_charge_512_8' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_1':
            if h5data == 'without_tresh':
                np.save(predict_where + 'dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
            elif h5data == 'with_tresh':
                np.save(predict_where + 'with_tresh_dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
            elif h5data == 'data_20':
                np.save(predict_where + '20_dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
            elif h5data == '512_sampled':
                np.save(predict_where + '5_dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
            elif h5data == 'data_250':
                np.save(predict_where + '250_dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
            elif h5data == 'with_n':
                np.save(predict_where + 'with_n_dresser_chair_weights_without_charge_512_1' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                        '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024_16':
            np.save(predict_where + 'dresser_chair_weights_without_charge_1024_16' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024_1':
            np.save(predict_where + 'dresser_chair_weights_without_charge_1024_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_2_without_charge_512_1':
            np.save(predict_where + 'dresser_chair_weights_without_charge_2_512_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_x' and weights_type == 'dresser_chair_without_charge_x_512_1':
            np.save(predict_where + 'dresser_chair_weights_without_charge_x_512_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge_z' and weights_type == 'dresser_chair_without_charge_z_512_1':
            np.save(predict_where + 'dresser_chair_weights_without_charge_z_512_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'sofa_table_without_charge_512_1':
            np.save(predict_where + 'sofa_table_without_charge_512_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'chair_6_512_1':
            np.save(predict_where + 'chair_6_without_charge_512_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_30':
            np.save(predict_where + 'dresser_chair_without_charge_512_1_30' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_375_1':
            np.save(predict_where + 'dresser_chair_without_charge_375_1' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_512':
            np.save(predict_where + 'dresser_chair_without_charge_512_1_512' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_2048':
            np.save(predict_where + 'dresser_chair_without_charge_512_1_2048' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_250':
            np.save(predict_where + 'dresser_chair_without_charge_512_1_2048' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_512':
            if h5data == 'without_chatrge':
                np.save(predict_where + 'dresser_chair_without_charge_512_1_512' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
            elif h5data == '512_sampled':
                np.save(predict_where + '5_dresser_chair_without_charge_512_1_512' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_2048':
            if h5data == '512_sanpled':
                np.save(predict_where + 'dresser_chair_without_charge_512_1_2048' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + 
                        '_nu_' + str(i), predict_2 )
            elif h5data == 'without_tresh':
                 np.save(predict_where + 'old_dresser_chair_without_charge_512_1_2048' +
                        '_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + 
                        '_nu_' + str(i), predict_2 )
                
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_64_1024':
            if h5data == '64_sampled':
                np.save(predict_where + 'dresser_chair_without_charge_64_1024' 
                        +'_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
            elif h5data == '64_slice':
                np.save(predict_where + 'slice_dresser_chair_without_charge_64_1024' 
                        +'_Shuffle_' +
                        'train_predict_' +   str(num_of_train_ex) + '_' 
                        + str(num_of_predict) + 
                        '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
                
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
           
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_32_1024':
            np.save(predict_where + 'dresser_chair_without_charge_32_1024' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_128_1024':
            np.save(predict_where + 'dresser_chair_without_charge_128_1024' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_256_1024':
            np.save(predict_where + 'dresser_chair_without_charge_256_1024' +'_Shuffle_' +
            'train_predict_' +   str(num_of_train_ex) + '_' 
            + str(num_of_predict) + 
            '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) + '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_old':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_old' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_x':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_x' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_y':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_y' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_n':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_n' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_ncx':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_ncx' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_ncz':
            np.save(predict_where + 'dresser_chair_without_charge_512_1024_ncz' +
                    '_Shuffle_' +
                    'train_predict_' +   str(num_of_train_ex) + '_' 
                    + str(num_of_predict) + 
                    '_kernel_' + str(kernel) +'_gamma_'+ str(gamma) +
                    '_nu_' + str(i), predict_2 )



        
      
        anom_ind_2 = predict_2 == -1
        not_anom_ind_2 = predict_2 == 1

        anom_ind_2 = anom_ind_2.T # SAVE
        not_anom_ind_2 = not_anom_ind_2.T # SAVE

        anom_2 = train_feature_without_shuffle[anom_ind_2]
        not_anom_2 = train_feature_without_shuffle[not_anom_ind_2]

        anom_2 = reducer.fit_transform(anom_2)
        not_anom_2 = reducer.fit_transform(not_anom_2)

        plt.scatter(anom_2[:, 0], 
            anom_2[:, 1],
            marker = '.',
            s = 5,
            color = 'r')

        plt.scatter(not_anom_2[:, 0],
            not_anom_2[:, 1],
            marker = '.',
            s = 5,
            color = 'g')

        
        if data_type == 'with_charge' and weights_type == 'Mg22':
            plt.title('Mg22 Weights ' + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_' +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
            
        elif data_type == 'with_charge' and weights_type == 'random':
            plt.title('Random Weights ' + str(nrand) + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Random_Weights_'+ str(nrand) +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
            
        elif data_type == 'without_charge' and weights_type == 'Mg22_without_charge':
            plt.title('Mg22 Weights  '  + data_type +  '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_'+ data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
            
        elif data_type == 'without_charge' and weights_type == 'random':
            plt.title('Random Weights ' + data_type +  str(nrand) + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Random_Weights_'+ str(nrand) + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_y' and weights_type == 'Mg22_without_charge_y':
            plt.title('Random Weights ' + data_type  + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_x' and weights_type == 'Mg22_without_charge_x':
            plt.title('Mg22 Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_z' and weights_type == 'Mg22_without_charge_z':
            plt.title('Mg22 Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'chair':
            plt.title('Chair Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_x' and weights_type == 'chair_without_charge_x':
            plt.title('Chair Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_y' and weights_type == 'chair_without_charge_y':
            plt.title('Chair Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_z' and weights_type == 'chair_without_charge_z':
            plt.title('Chair Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'chair_2_without_charge':
            plt.title('Chair Weights 2 ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_2_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_x' and weights_type == 'chair_2_without_charge_x':
            plt.title('Chair Weights 2 ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_2_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
       
        elif data_type == 'without_charge_z' and weights_type == 'Mg22_without_charge_z_10':
            plt.title('Mg22 Weights ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Mg22_Weights_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
            
        elif data_type == 'without_charge_y' and weights_type == 'chair_2_without_charge_y':
            plt.title('Chair Weights 2 ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_2_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_z' and weights_type == 'chair_2_without_charge_z':
            plt.title('Chair Weights 2 ' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'Chair_Weights_2_' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair':
            plt.title('dresser chair Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'sofa_table':
            plt.title('sofa table  Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'sofa_table_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_x' and weights_type == 'dresser_chair_without_charge_x':
            plt.title('dresser chair  Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024':
            plt.title('dresser chair 1024 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_1024_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512':
            plt.title('dresser chair 512 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_256':
            plt.title('dresser chair 256 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_256_Weights' + data_type +  '_Shuffle_train_predict' + str(num_of_train_ex) + 
                '_' + str(num_of_predict) + 
                '_kernel_' + str(kernel) + 
                '_gamma_'+ str(gamma) + 
                '_nu: ' + str(i)  +
                '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_big':
            plt.title('dresser chair 512 big Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_big_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_big_2':
            plt.title('dresser chair 512 big_2 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_big_2_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_16':
            plt.title('dresser chair 512 16 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_16_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_2048_32':
            plt.title('dresser chair 2048 32 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_2048_32_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_8':
            plt.title('dresser chair 512 8 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_8_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_512_1':
            if h5data == 'without_tresh':
                plt.title('dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == 'with_tresh':
                plt.title('with tresh dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'with_tresh_dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == 'data_20':
                plt.title('data_20 dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + '20_dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == '512_sampled':
                plt.title('data_5 dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + '5_dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == 'data_250':
                plt.title('data_250 dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + '250_dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == 'with_n':
                plt.title('with_n dresser chair 512 1 Weights' + data_type + 
                          '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'with_n_dresser_chair_512_1_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024_16':
            plt.title('dresser chair 1024 16 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_1024_16_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_without_charge_1024_1':
            plt.title('dresser chair 1024 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_1024_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_2_without_charge_512_1':
            plt.title('dresser chair_2 512 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_2_512_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_x' and weights_type == 'dresser_chair_without_charge_x_512_1':
            plt.title('dresser chair 512 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_2_512_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge_z' and weights_type == 'dresser_chair_without_charge_z_512_1':
            plt.title('dresser chair 512 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'sofa_table_without_charge_512_1':
            plt.title('sofa table 512 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'sofa_table_512_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'chair_6_512_1':
            plt.title('chair_6 512 1 Weights' + data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'chair_6_512_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_30':
            plt.title('dresser_chair 512 1 30 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1_30_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_375_1':
            plt.title('dresser_chair 375 1 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_375_1_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_512':
            plt.title('dresser_chair 512 1 512 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1_512_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1_2048':
            plt.title('dresser_chair 512 1 2048 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1_2048_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_250':
            plt.title('dresser_chair 250 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_250_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_512':
            if h5data == 'without_charge':
                plt.title('dresser_chair 512 512 Weights' + 
                          data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                      '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'dresser_chair_512_512_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                        '_' + str(num_of_predict) + 
                        '_kernel_' + str(kernel) + 
                        '_gamma_'+ str(gamma) + 
                        '_nu: ' + str(i)  +
                        '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == '512_sampled':
                plt.title('data 5 dresser_chair 512 512 Weights' + 
                          data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                      '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + '5_dresser_chair_512_512_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                        '_' + str(num_of_predict) + 
                        '_kernel_' + str(kernel) + 
                        '_gamma_'+ str(gamma) + 
                        '_nu: ' + str(i)  +
                        '_anom: ' + str(anom_2.shape[0]) +  '.png')
                
                
                
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_2048':
            if h5data == '512_sampled':
                plt.title('dresser_chair 512 2048 Weights' + 
                          data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                      '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'dresser_chair_512_2048_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                        '_' + str(num_of_predict) + 
                        '_kernel_' + str(kernel) + 
                        '_gamma_'+ str(gamma) + 
                        '_nu: ' + str(i)  +
                        '_anom: ' + str(anom_2.shape[0]) +  '.png')
            elif h5data == 'without_tresh':
                plt.title('old dresser_chair 512 2048 Weights' + 
                          data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                      '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'old_dresser_chair_512_2048_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                        '_' + str(num_of_predict) + 
                        '_kernel_' + str(kernel) + 
                        '_gamma_'+ str(gamma) + 
                        '_nu: ' + str(i)  +
                        '_anom: ' + str(anom_2.shape[0]) +  '.png')
                
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024':
          
            plt.title('dresser_chair 512 1024 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                      '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                        '_' + str(num_of_predict) + 
                        '_kernel_' + str(kernel) + 
                        '_gamma_'+ str(gamma) + 
                        '_nu: ' + str(i)  +
                        '_anom: ' + str(anom_2.shape[0]) +  '.png')
           
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_64_1024':
            if h5data == '64_sampled':
                plt.title('dresser_chair 64 1024 Weights' + 
                          data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                          '; Anomalies: ' + str(anom_2.shape[0]))
                plt.show()

                plt.savefig(umap_where + 'dresser_chair_64_1024_Weights' + data_type + 
                            '_Shuffle_train_predict' + str(num_of_train_ex) + 
                            '_' + str(num_of_predict) + 
                            '_kernel_' + str(kernel) + 
                            '_gamma_'+ str(gamma) + 
                            '_nu: ' + str(i)  +
                            '_anom: ' + str(anom_2.shape[0]) +  '.png')
            
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_32_1024':
            plt.title('dresser_chair 32 1024 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_32_1024_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_128_1024':
            plt.title('dresser_chair 128 1024 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_128_1024_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_256_1024':
            plt.title('dresser_chair 256 1024 Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_256_1024_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_old':
            plt.title('dresser_chair 512 1024 old Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_old_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_x':
            plt.title('dresser_chair 512 1024 x Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_x_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_y':
            plt.title('dresser_chair 512 1024 y Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_y_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_n':
            plt.title('dresser_chair 512 1024 n Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_n_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_ncx':
            plt.title('dresser_chair 512 1024 ncx Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_ncx_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')
        elif data_type == 'without_charge' and weights_type == 'dresser_chair_512_1024_ncz':
            plt.title('dresser_chair 512 1024 ncz Weights' + 
                      data_type + '; Shuffle; Examples: ' + str(num_of_predict) +
                  '; Anomalies: ' + str(anom_2.shape[0]))
            plt.show()

            plt.savefig(umap_where + 'dresser_chair_512_1024_ncz_Weights' + data_type + 
                        '_Shuffle_train_predict' + str(num_of_train_ex) + 
                    '_' + str(num_of_predict) + 
                    '_kernel_' + str(kernel) + 
                    '_gamma_'+ str(gamma) + 
                    '_nu: ' + str(i)  +
                    '_anom: ' + str(anom_2.shape[0]) +  '.png')





        


     