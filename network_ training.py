# Library for 
import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l2
import keras
from sklearn.model_selection import PredefinedSplit
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import keras.backend as K

# Setp up seed and training environment  # Note that training with GPU is not reproducible, but fixing seeds can reduce randomness.
from tensorflow import set_random_seed
import random as rn
# set seeds
keras.initializers.glorot_normal(seed=1)
np.set_printoptions(suppress=True)
set_random_seed(1)
np.random.seed(1)
rn.seed(1)
os.environ['PYTHONHASHSEED'] = '0'

# training with GPU setting
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config, graph=tf.get_default_graph())
K.set_session(sess)


# DenseNet implementation, from  https://github.com/seasonyc/densenet, output layers are adjusted for the thesis problem
def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    """
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST    
        dense_blocks : amount of dense blocks that will be created (default: 3)    
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in  block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)
    Returns:
        Model        : A Keras model instance
    """
    
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    print(dense_layers)
    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2
    
    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')
    
    # Initial convolution layer
    x = Conv2D(nb_channels, (3,3), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    
    # Building dense blocks
    for block in range(dense_blocks):
        
        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(25, activation='sigmoid', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'
        
    if bottleneck:
        model_name = model_name + 'b'
        
    if compression < 1.0:
        model_name = model_name + 'c'
        
    return Model(img_input, x, name=model_name), model_name


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """
    
    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """
    
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

# Functions for conducting grid-search on hyperparameters
def create_model(BL,GR, DR, CR, WD, D, DL, DB):
    K.clear_session()
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model, name = DenseNet((27,27,1), nb_classes = 25, bottleneck=BL,
                       growth_rate=GR, dropout_rate=DR, compression=CR, dense_blocks= DB, 
                       weight_decay=WD, depth=D, dense_layers= DL)
    model.compile(loss="binary_crossentropy", optimizer=adam)
    return model

def grid_search(x_train, x_vlad, y_train, y_vlad):
    my_test_fold = []
	# Each model only test on vlidation set, -1 means it will not be tested
    for i in range(len(x_train)):
	    my_test_fold.append(-1)
    for i in range(len(x_vlad)):
	    my_test_fold.append(0)

    ps = PredefinedSplit(test_fold=my_test_fold)
    new_x = np.concatenate((x_train, x_vlad), axis=0)
    new_y = np.concatenate((y_train, y_vlad), axis=0)
    M =KerasClassifier(build_fn=create_model, verbose=0, nb_epoch=10)
	# The set of parameters defined in the thesis
    batch_size = [32]
    GR = [ 4, 8 ,12]
    DR =[ 0.5]
    CR = [0.5, 1]
    WD = [1e-4]
    DB = [1]
    D = [20,30, 40]
    DL = [-1]
    BL = [False, True]
    param_grid = dict(batch_size=batch_size, GR=GR, DR= DR, CR=CR, WD=WD, D=D, DL=DL, BL=BL, DB=DB)
    ftwo_scorer = make_scorer(f1_score, average='micro')
    grid = GridSearchCV(estimator=M, param_grid=param_grid, n_jobs=1, cv=ps, scoring=ftwo_scorer, refit= False)
    grid_result = grid.fit(new_x, new_y)
	# Print the result for the grid-search and the best  parameter
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
	       print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def main():
	# Load training data
    with open('train_valid.pickle', 'rb') as f:
	    x_train, y_train, x_vlad, y_vlad = pickle.load(f)

 	# Grid-search for hyperparameters
    grid_result = grid_search(x_train, x_vlad, y_train, y_vlad )
    para = grid_result.best_params_
    GR = para['GR']
    DB = para['DB']
    DR = 0.5 # use drop out layer to prevent overfitting
    CR = para['CR']
    WD= para['WD']
    D = para['D']
    BL = para['BL']
    DL = -1 # same number of layers in each block

	# Training 
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model, name = DenseNet((27,27,1), nb_classes = 25, bottleneck=BL, dense_blocks = DB,
	                       growth_rate=GR, dropout_rate=DR, compression=CR, 
	                       weight_decay=WD, depth=D, dense_layers= DL)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
    new_x = np.concatenate((x_train, x_vlad), axis=0)
    new_y = np.concatenate((y_train, y_vlad), axis=0)
    H = model.fit(new_x, new_y, epochs=30, batch_size =32)
    model.save('denseModel.h5')

if __name__== "__main__":
  main()
